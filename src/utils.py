"""
CERTified Edit Distance defense (CERT-ED) authors authored this file

ChatGPT and/or Copilot are used in generating scaffolding code for this file
"""
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import zipfile
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests
import scipy.special as sp
import torch
from bs4 import BeautifulSoup
from datasets import ClassLabel
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict
from datasets import load_dataset as huggingface_load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from certification import (
    PerturbationTokenizer,
    SmoothedClassifierWrapper,
    perturbation_tokenizers,
)


class PerturbedDataset(Dataset):
    def __init__(
        self,
        data: DatasetDict,
        perturbation_tokenizer: PerturbationTokenizer,
        max_length: int = 256,
        padding="longest",
    ):
        self.data = data
        self.padding = padding
        self.max_length = max_length
        self.perturbation_tokenizer = perturbation_tokenizer
        self.perturbation_on = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        old_perturbation_on = self.perturbation_tokenizer.perturbation_on
        self.perturbation_tokenizer.perturbation_on = self.perturbation_on

        text = self.data[idx]["text"]
        label = self.data[idx]["label"]
        encoded = self.perturbation_tokenizer(
            text,
            truncation=True,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(label)
        self.perturbation_tokenizer.perturbation_on = old_perturbation_on
        return encoded


def get_device(use_gpu):
    """Get device (CUDA if available and requested, else CPU)."""
    if use_gpu and torch.cuda.is_available():
        logging.info(f"Using CUDA device: {torch.cuda.current_device()}")
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    logging.info("Using CPU.")
    return torch.device("cpu")


def load_checkpoint(config):
    """
    Load the checkpoint based on the config. If a specific checkpoint path is given, use it.
    Otherwise, try to find the checkpoint in the default checkpoint directory.
    """
    checkpoint_path = config["load_checkpoint"]

    # Handle case where "best" is specified
    if checkpoint_path == "best":
        checkpoint_path = os.path.join(config["checkpoint_dir"], "best_checkpoint.pth")

    # Handle case where an epoch number is specified
    elif re.match(r"^\d+$", checkpoint_path):
        epoch_num = int(checkpoint_path)
        checkpoint_path = os.path.join(
            config["checkpoint_dir"], f"epoch_{epoch_num}_checkpoint.pth"
        )

    # Handle case where "last" is specified
    elif checkpoint_path == "last":
        all_files = [
            f
            for f in os.listdir(config["checkpoint_dir"])
            if os.path.isfile(os.path.join(config["checkpoint_dir"], f))
        ]
        epoch_files = sorted([f for f in all_files if "epoch_" in f])
        if epoch_files:
            last_epoch_file = epoch_files[-1]
            checkpoint_path = os.path.join(config["checkpoint_dir"], last_epoch_file)
        else:
            raise FileNotFoundError(
                f"No epoch files found in {config['checkpoint_dir']}"
            )

    # Handle the case where the checkpoint_path is actually a path
    elif not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(config["checkpoint_dir"], checkpoint_path)

    # Check if the final resolved path exists or not
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    return torch.load(checkpoint_path, map_location=torch.device("cpu"))


def load_model(config, device, num_labels=2):
    model_name = config["model"]["type"]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model.to(device)

    # Load state_dict from checkpoint if specified
    if "load_checkpoint" in config:
        checkpoint = load_checkpoint(config)
        model.load_state_dict(checkpoint["model_state_dict"])

    return model


def load_perturbation_tokenizer(config):
    perturbation_class = perturbation_tokenizers[config["perturbation"]]
    perturbation_tokenizer = perturbation_class(
        tokenizer=config["model"]["type"], **config.get("perturbation_args", {})
    )
    return perturbation_tokenizer


def load_optimizer(config, model):
    optimizer_class = getattr(torch.optim, config["optimizer"])
    if config["optimizer"] == "AdamW":
        # Extract weight decay from optimizer_args or default to 0.01
        weight_decay = config.get("optimizer_args", {}).pop("weight_decay", 0.01)
        # Separate out parameters that shouldn't have weight decay applied
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(
            optimizer_grouped_parameters, **config.get("optimizer_args", {})
        )
    else:
        optimizer = optimizer_class(
            model.parameters(), **config.get("optimizer_args", {})
        )

    # Load state_dict from checkpoint if specified
    if "load_checkpoint" in config:
        checkpoint = load_checkpoint(config)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return optimizer


def load_grad_scaler(config, peturbation_tokenizer):
    if config["mask_grad_scale"] is None and config["grad_clip"] is None:
        return None
    mask_token_id = peturbation_tokenizer.tokenizer.mask_token_id

    def grad_scaler(model):
        embed = model.get_input_embeddings()
        if config["mask_grad_scale"] is not None:
            embed.weight.grad[mask_token_id] *= config["mask_grad_scale"]
        if config["grad_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    return grad_scaler


def load_scheduler(config, optimizer, total_steps=None):
    if "scheduler" in config:
        if config["scheduler"] == "linear_schedule_with_warmup":
            # Ensure total_steps is provided for this scheduler
            if total_steps is None:
                raise ValueError(
                    "total_steps must be provided for get_linear_schedule_with_warmup"
                )

            # Infer num_warmup_steps and num_training_steps
            warmup_fraction = config.get("scheduler_args", {}).pop(
                "warmup_fraction", 0.05
            )
            num_warmup_steps = int(total_steps * warmup_fraction)
            num_training_steps = total_steps

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **config.get("scheduler_args", {}),
            )
        else:
            # Use the original torch.optim.lr_scheduler
            scheduler_class = getattr(torch.optim.lr_scheduler, config["scheduler"])
            scheduler = scheduler_class(optimizer, **config.get("scheduler_args", {}))

        # Load state_dict from checkpoint if specified
        if "load_checkpoint" in config:
            checkpoint = load_checkpoint(config)
            if (
                "scheduler_state_dict" in checkpoint
                and checkpoint["scheduler_state_dict"]
            ):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return scheduler
    return None


def load_components(config):
    device = get_device(config["use_gpu"])
    train_data = load_dataset(**config["dataset"])["train"]
    if config["sample_size"] is not None:
        train_data = train_data.train_test_split(
            train_size=config["sample_size"],
            stratify_by_column="label",
            seed=config["seed"],
        )["train"]
    train_data = train_data.train_test_split(
        test_size=config["valid_size"],
        stratify_by_column="label",
        seed=config["seed"],
    )
    test_data = load_dataset(**config["dataset"])["test"]

    num_labels = len(set(train_data["train"]["label"]))
    perturbation_tokenizer = load_perturbation_tokenizer(config)
    model = load_model(config=config, device=device, num_labels=num_labels)
    smoothed_model = SmoothedClassifierWrapper(
        model=model,
        perturbation_tokenizer=perturbation_tokenizer,
        device=device,
    )
    optimizer = load_optimizer(config, model)
    grad_scaler = load_grad_scaler(config, perturbation_tokenizer)
    loss_function = torch.nn.CrossEntropyLoss().to(device)

    if "load_checkpoint" in config:
        checkpoint = load_checkpoint(config)
    else:
        checkpoint = None

    train_dataset = PerturbedDataset(
        train_data["train"], perturbation_tokenizer=perturbation_tokenizer
    )
    valid_dataset = PerturbedDataset(
        train_data["test"], perturbation_tokenizer=perturbation_tokenizer
    )
    test_dataset = PerturbedDataset(
        test_data, perturbation_tokenizer=perturbation_tokenizer
    )

    num_workers = config["num_workers"]
    batch_size = config["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=DataCollatorWithPadding(perturbation_tokenizer.tokenizer),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=DataCollatorWithPadding(perturbation_tokenizer.tokenizer),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=DataCollatorWithPadding(perturbation_tokenizer.tokenizer),
    )
    total_steps = len(train_loader) * config["max_epochs"]
    scheduler = load_scheduler(config, optimizer, total_steps=total_steps)

    return {
        "perturbation_tokenizer": perturbation_tokenizer,
        "model": model,
        "smoothed_model": smoothed_model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "grad_scaler": grad_scaler,
        "loss_function": loss_function,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "num_labels": num_labels,
        "device": device,
        "checkpoint": checkpoint,
    }


def setup_tqdm(total, desc=None):
    is_jupyter = "ipykernel" in sys.modules
    is_terminal = sys.stdout.isatty()
    tqdm_params = {
        "total": total,
        "desc": desc,
        "disable": not (is_terminal or is_jupyter),
        "dynamic_ncols": True,
        "file": sys.stdout,
    }
    if tqdm_params["disable"]:
        tqdm_params["bar_format"] = "{l_bar}{bar:10}{r_bar}{bar:-10b}"

    return tqdm_params


def remove_oldest_files(base_dir, keep=3, exclude_files=None):
    all_files = [
        f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))
    ]
    if exclude_files is not None:
        all_files = [f for f in all_files if f not in exclude_files]
    all_files.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir, x)))

    while len(all_files) > keep:
        os.remove(os.path.join(base_dir, all_files[0]))
        del all_files[0]


def download_google_drive_file(file_id, destination, bypass=False):
    URL = f"https://docs.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(URL, stream=True)

    if bypass:
        # Parse for a confirmation token
        soup = BeautifulSoup(response.text, "html.parser")
        confirm_form = soup.find("form", {"id": "download-form"})

        if confirm_form:
            confirm_form_action = confirm_form["action"]
            parsed_url = urlparse(confirm_form_action)
            parsed_query = parse_qs(parsed_url.query)
            confirm_token = parsed_query.get("confirm")[0]

            if confirm_token:
                params = {"id": file_id, "confirm": confirm_token}
                response = session.get(URL, params=params, stream=True)

    # Write to a zip file
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def unzip_file(zip_filepath, dest_dir):
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(dest_dir)


def load_dataset(path, download=False, *args, **kwargs):
    advbench_datasets = [
        "advbench/amazon_lb",
        "advbench/assassin",
        "advbench/CGFake",
        "advbench/EDENCE",
        "advbench/enron",
        "advbench/FAS",
        "advbench/HSOL",
        "advbench/jigsaw",
        "advbench/LUN",
        "advbench/satnews",
    ]
    if path in advbench_datasets:
        dataset_name = path.split("/")[-1]
        dataset_path = os.path.join(os.getcwd(), dataset_name)
        if not os.path.exists(dataset_path):
            data_folder = os.path.expanduser("~/.cache/advbench")
            dataset_path = os.path.join(data_folder, dataset_name)

            if not os.path.exists(dataset_path):
                if download:
                    os.makedirs(data_folder, exist_ok=True)
                    file_id = "1BOyyblqrBAZ4qgGbZOuBe85sxn8rX6Xg"
                    zip_name = "compressed.zip"
                    destination = os.path.join(data_folder, zip_name)
                    download_google_drive_file(file_id, destination, bypass=True)
                    unzip_file(destination, data_folder)
                else:
                    raise FileNotFoundError(
                        f"The dataset {path} does not exist at {dataset_path}"
                    )
        train_dataset = load_csv_dataset(os.path.join(dataset_path, "train.csv"))
        test_dataset = load_csv_dataset(os.path.join(dataset_path, "dev.csv"))
        return DatasetDict({"train": train_dataset, "test": test_dataset})
    elif "attack" in path:
        train_dataset = load_attacked_csv_dataset(path, *args, **kwargs)
        test_dataset = load_attacked_csv_dataset(path, *args, **kwargs)
        return DatasetDict({"train": train_dataset, "test": test_dataset})
    else:
        return huggingface_load_dataset(path, *args, **kwargs)


def read_csv_to_dict(filepath):
    with open(filepath, mode="r") as file:
        reader = csv.DictReader(file)
        return {name: [row[name] for row in reader] for name in reader.fieldnames}


def remove_brackets(text):
    # Pattern to match words enclosed in double square brackets
    pattern = r"\[\[([^\[\]]+)\]\]"
    # Replace the matched pattern with just the word inside the brackets
    result = re.sub(pattern, r"\1", text)
    return result


# "original_text","perturbed_text","original_score","perturbed_score","original_output","perturbed_output","ground_truth_output","num_queries","result_type"
def load_attacked_csv_dataset(path, perturbed):
    if perturbed:
        text_col = "perturbed_text"
    else:
        text_col = "original_text"

    label_col = "ground_truth_output"
    data = pd.read_csv(path)[[text_col, label_col]]
    data[text_col] = data[text_col].apply(remove_brackets)
    data = data.rename(columns={text_col: "text", label_col: "label"}).to_dict(
        orient="list"
    )
    num_classes = len(set(data["label"]))
    dataset = HuggingFaceDataset.from_dict(data)
    dataset.features["label"] = ClassLabel(num_classes=num_classes)
    return dataset


def load_csv_dataset(path):
    data = pd.read_csv(path).to_dict(orient="list")
    num_classes = len(set(data["label"]))
    dataset = HuggingFaceDataset.from_dict(data)
    dataset.features["label"] = ClassLabel(num_classes=num_classes)
    return dataset


def subsample_dataset(dataset, sample_size, seed=42):
    """Subsample a Hugging Face dataset deterministically based on a seed.

    Args:
        dataset (datasets.Dataset): The Hugging Face dataset to subsample.
        seed (int): The random seed for reproducibility.
        sample_size (int): The number of samples to draw.

    Returns:
        datasets.Dataset: The subsampled dataset.
    """

    # Initialize the random number generator with the seed
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected_indices = indices[:sample_size]
    subsampled_dataset = dataset.select(selected_indices)
    return selected_indices, subsampled_dataset


def hash_dict(d):
    serialized_data = json.dumps(d, sort_keys=True).encode("utf-8")
    full_hash = int(hashlib.sha256(serialized_data).hexdigest(), 16)
    return full_hash


def l0_distance(list1, list2):
    """
    Calculate the L0 distance between two lists.

    Args:
        list1 (list): The first list.
        list2 (list): The second list.

    Returns:
        int: The L0 distance between the two lists.
    """
    return sum(1 for x, y in zip(list1, list2) if x != y) + abs(len(list1) - len(list2))


def edit_distance(list1, list2):
    """
    Calculate the Edit (Levenshtein) distance between two lists.

    Args:
        list1 (list): The first list.
        list2 (list): The second list.

    Returns:
        int: The Edit distance between the two lists.
    """
    m, n = len(list1), len(list2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # Min. operations = j
            elif j == 0:
                dp[i][j] = i  # Min. operations = i
            elif list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i][j - 1],  # Insert
                    dp[i - 1][j],  # Remove
                    dp[i - 1][j - 1],  # Replace
                )
    return dp[m][n]


def combln(n, k) -> float:
    return sp.gammaln(n + 1) - sp.gammaln(k + 1) - sp.gammaln(n - k + 1)


def edit_distance_volume(radius, input_size, vocab_size, log=False):
    if log:
        log_out = -np.inf
        for i in range(radius + 1):
            log_inner = -np.inf
            for j in range(i - radius, radius + 1):
                log_inner = np.logaddexp(log_inner, combln(input_size + j, i))

            log_out = np.logaddexp(log_out, i * np.log(vocab_size - 1) + log_inner)
        return log_out / np.log(10)
    else:
        out = 0
        for i in range(radius + 1):
            inner = 0
            for j in range(i - radius, radius + 1):
                inner += sp.comb(input_size + j, i, exact=True)
            out += ((vocab_size - 1)**i) * inner
        return out


def l0_distance_volume(radius, input_size, vocab_size, log=False):
    if log:
        log_out = -np.inf
        for i in range(radius + 1):
            log_out = np.logaddexp(log_out, combln(input_size, i) + i * np.log(vocab_size - 1))
        return log_out / np.log(10)
    else:
        out = 0
        for i in range(radius + 1):
            out += sp.comb(input_size, i) * (vocab_size - 1)**radius
        return out
