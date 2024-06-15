"""
CERTified Edit Distance defense (CERT-ED) authors authored this file

ChatGPT and/or Copilot are used in generating scaffolding code for this file
"""
import logging
import os
from math import ceil

import pandas as pd
import torch
from tqdm import tqdm
from tqdm.auto import tqdm

from certification import SmoothedClassifierWrapper, certified_predictions_to_dataframe
from utils import PerturbedDataset, load_components, setup_tqdm


def evaluate_batch(
    inputs: str,
    smoothed_model: SmoothedClassifierWrapper,
    pred_num_samples: int,
    pred_kwargs: dict = {},
    cr_num_samples=0,
    cr_kwargs: dict = {},
    batch_size=32,
):
    all_certified_predictions = []
    for text in inputs:
        certified_prediction = smoothed_model.certify(
            text=text,
            pred_num_samples=pred_num_samples,
            cr_num_samples=cr_num_samples,
            batch_size=batch_size,
            pred_kwargs=pred_kwargs,
            cr_kwargs=cr_kwargs,
        )
        all_certified_predictions.append(certified_prediction)
    return all_certified_predictions


def evaluate(
    model: torch.nn.Module,
    dataset: PerturbedDataset,
    pred_num_samples: int,
    pred_kwargs: dict = {},
    cr_num_samples: int = 0,
    cr_kwargs: dict = {},
    batch_size: int = 32,
    device: torch.DeviceObjType = "cpu",
    checkpoint_interval: float = None,
    checkpoint_path: str = None,
):
    all_preds = []
    all_labels = []
    all_inputs = []
    # Existing code for initializing your models and other variables
    perturbation_tokenizer = dataset.perturbation_tokenizer
    data_dict = dataset.data
    # Initialize the counter for processed batches and calculate checkpoint frequency
    processed = 0
    total = len(data_dict)

    model.eval()
    smoothed_model = SmoothedClassifierWrapper(
        model=model,
        perturbation_tokenizer=perturbation_tokenizer,
        device=device,
    )
    with torch.no_grad():
        tqdm_params = setup_tqdm(total=total, desc="Evaluation progress")
        with tqdm(**tqdm_params) as progress_bar:
            for data in data_dict:
                text, label = data["text"], data["label"]
                preds = evaluate_batch(
                    inputs=[text],
                    smoothed_model=smoothed_model,
                    pred_num_samples=pred_num_samples,
                    pred_kwargs=pred_kwargs,
                    cr_num_samples=cr_num_samples,
                    cr_kwargs=cr_kwargs,
                    batch_size=batch_size,
                )
                all_preds.append(preds[0])
                all_labels.append(label)
                all_inputs.append(text)

                processed += 1
                if (checkpoint_interval is not None and processed % checkpoint_interval == 0):
                    df = certified_predictions_to_dataframe(all_preds)
                    df["label"] = all_labels
                    if checkpoint_path is not None:
                        df.to_csv(checkpoint_path, index=False)
                    accuracy = sum(
                        p.pred == t for p, t in zip(all_preds, all_labels) if p is not None
                    ) / len(all_labels)
                    logging.info(
                        f"Certification Accuracy ({processed}/{total}): {accuracy:<7.2%})"
                    )
                progress_bar.update(1)
    # Final construction
    accuracy = sum(
        p.pred == t for p, t in zip(all_preds, all_labels) if p is not None
    ) / len(all_labels)
    df = certified_predictions_to_dataframe(all_preds)
    df["label"] = all_labels
    df["input"] = all_inputs
    return accuracy, df


def certify_model(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    load_df = config["load_df"]

    pred_dir = config["pred_dir"]
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    
    if "dataset" in config:
        config["model_config"]["dataset"] = config["dataset"]

    components = load_components(config["model_config"])
    model = components["model"]
    test_dataset = components["test_dataset"]

    df = None
    preds_path = os.path.join(config["pred_dir"], "certified_predictions.csv")
    tmp_preds_path = os.path.join(config["pred_dir"], "certified_predictions_tmp.csv")

    if load_df:
        try:
            df = pd.read_csv(preds_path)
        except:
            logging.warning(
                "Failed to load certified predictions. Default to re-run certify."
            )

    if df is None:
        test_size = config["test_size"]
        if test_size is not None:
            test_dataset.data = test_dataset.data.train_test_split(
                test_size=config["test_size"],
                seed=config["seed"],
                stratify_by_column="label",
            )["test"]

        device = components["device"]
        acc, df = evaluate(
            model=model,
            dataset=test_dataset,
            pred_num_samples=config["pred_num_samples"],
            pred_kwargs=config["pred_kwargs"],
            cr_num_samples=config["cr_num_samples"],
            cr_kwargs=config["cr_kwargs"],
            batch_size=config["batch_size"],
            device=device,
            checkpoint_interval=config["checkpoint_interval"],
            checkpoint_path=tmp_preds_path,
        )
    else:
        perturbation_tokenizer = components["perturbation_tokenizer"]
        pred_label_counts = df[
            [f"label_{i}_count" for i in range(components["num_labels"])]
        ].to_numpy()
        cr_label_counts = df[
            [f"cr_label_{i}_count" for i in range(components["num_labels"])]
        ].to_numpy()
        inputs = df["input"].to_numpy()
        preds, pred_pvals, cr_preds, certified_radii = [], [], [], []
        for input, pred_counts, cr_counts in zip(
            inputs, pred_label_counts, cr_label_counts
        ):
            pred, pred_pval = perturbation_tokenizer.predict(
                input, pred_counts, **config["pred_kwargs"]
            )
            cr_pred, certified_radius = perturbation_tokenizer.certified_radius(
                input, cr_counts, **config["cr_kwargs"]
            )
            preds.append(pred), pred_pvals.append(pred_pval), cr_preds.append(
                cr_pred
            ), certified_radii.append(certified_radius)
        df["pred"], df["pred_pval"], df["cr_pred"], df["certified_radius"] = (
            preds,
            pred_pvals,
            cr_preds,
            certified_radii,
        )
        acc = (df["pred"] == df["label"]).mean()

    message = f"Certification Accuracy: {acc:<7.2%}"
    logging.info(message)
    logging.info("Certification completed.")
    df.to_csv(preds_path, index=False)
