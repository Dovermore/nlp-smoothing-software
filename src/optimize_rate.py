"""
ChatGPT and/or Copilot are used in generating scaffolding code for this file
"""

import logging
import os
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from utils import (
    PerturbedDataset,
    load_components,
    setup_tqdm,
    golden_section_search,
)
from certify import evaluate


def optimize_rate(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if "dataset" in config:
        config["model_config"]["dataset"] = config["dataset"]

    # Disable this for now
    #config["model_config"]["output_dir"] = config["output_dir"]

    components = load_components(config["model_config"])
    model = components["model"]

    if config["optim_split"] == "train":
        optim_dataset = components["train_dataset"]
    elif config["optim_split"] == "test":
        optim_dataset = components["test_dataset"]
    else:
        optim_dataset = components["valid_dataset"]
    perturbation_tokenizer = components["perturbation_tokenizer"]

    # Fetch the search log path from VarDelOptim's agent_dir
    search_log_path = os.path.join(
        perturbation_tokenizer.agent_dir, "best_search_log.csv"
    )
    tmp_search_log_path = os.path.join(
        perturbation_tokenizer.agent_dir, "tmp_search_log.csv"
    )

    device = components["device"]
    optim_size = config["optim_size"]
    certified_accuracy_threshold = config["certified_accuracy_threshold"]

    # Extract bins and initialize results
    bins = perturbation_tokenizer.f_del.bins
    bin_idxs = list(range(len(bins) - 1))
    bin_results = [
        {
            "best_value": np.nan,
            "best_radius": np.nan,
        }
        for _ in bin_idxs
    ]

    tqdm_params = setup_tqdm(total=len(bins) - 1, desc="Optimizing bins")
    # always enable
    tqdm_params["disable"] = False
    tqdm_params["leave"] = True
    tqdm_params["initial"] = -1

    with tqdm(**tqdm_params) as progress_bar:
        search_logs = []
        for bin_idx in bin_idxs[::-1]:
            bin_start, bin_end = bins[bin_idx], bins[bin_idx + 1]
            # Sample from the training dataset based on bin range
            sampled_data_dict = optim_dataset.data.filter(
                lambda x: bin_start <= len(x["text"].split(" ")) < bin_end
            )
            sampled_data_dict = sampled_data_dict.shuffle().select(
                range(min(len(sampled_data_dict), optim_size))
            )
            progress_bar.set_description(
                f"Optimizing: {bin_idx} - {bin_start}-{bin_end} ({len(sampled_data_dict)})", refresh=False,
            )
            progress_bar.update(1)

            min_length = min(len(x["text"].split(" ")) for x in sampled_data_dict)
            max_length = max(len(x["text"].split(" ")) for x in sampled_data_dict)

            def evaluate_bin(mapping_value):
                # Update the perturbation tokenizer for the bin
                perturbation_tokenizer.f_del.values = [mapping_value] * (len(bins) - 1)

                # Create a PerturbedDataset
                perturbed_dataset = PerturbedDataset(
                    sampled_data_dict, perturbation_tokenizer=perturbation_tokenizer
                )

                # Evaluate and get certification results
                acc, df = evaluate(
                    model=model,
                    dataset=perturbed_dataset,
                    pred_num_samples=config["pred_num_samples"],
                    pred_kwargs=config["pred_kwargs"],
                    cr_num_samples=config["cr_num_samples"],
                    cr_kwargs=config["cr_kwargs"],
                    batch_size=config["batch_size"],
                    device=device,
                    verbose=1,
                    warning_as_exception=False,
                )
                df["certified_radius"] = np.where(
                    (df["label"] == df["cr_pred"]) & (df["label"] == df["pred"]),
                    df["certified_radius"],
                    -df["certified_radius"],
                )

                # Compute certified accuracy at the largest radius meeting the threshold
                largest_radius = 0
                certified_accuracy_of_radius = 0
                for radius_threshold in sorted(df["certified_radius"].unique()):
                    certified_accuracy = (df["certified_radius"] >= radius_threshold).mean()
                    if certified_accuracy < certified_accuracy_threshold:
                        break
                    largest_radius = radius_threshold
                    certified_accuracy_of_radius = certified_accuracy
                mean_radius = df["certified_radius"].mean()
                return largest_radius, mean_radius, certified_accuracy_of_radius

            best_mapping_value, best_radius, search_log = golden_section_search(
                # best_mapping_value, best_radius, search_log = ternary_search(
                f=evaluate_bin,
                low=min_length * 0.01,
                high=max_length * 0.3,
                tolerance=0.1,
                log_results=True,
            )

            bin_results[bin_idx] = {
                "best_value": best_mapping_value,
                "best_radius": best_radius,
            }
            # Update the perturbation tokenizer with the optimized mapping values
            optimized_values = [bin_result["best_value"] for bin_result in bin_results]
            perturbation_tokenizer.f_del.values = optimized_values
            perturbation_tokenizer.save_agent("tmp")

            search_log["bin_idx"] = bin_idx
            search_log["bin_start"] = bin_start
            search_log["bin_end"] = bin_end
            search_logs.append(search_log)
            combined_log = pd.concat(search_logs, ignore_index=True).set_index("bin_idx")
            combined_log.to_csv(tmp_search_log_path, index=True)
        progress_bar.update(1)

    # Update the perturbation tokenizer with the optimized mapping values
    optimized_values = [bin_result["best_value"] for bin_result in bin_results]
    perturbation_tokenizer.f_del.values = optimized_values
    perturbation_tokenizer.save_agent("best")

    combined_log = pd.concat(search_logs, ignore_index=True).set_index("bin_idx")
    combined_log.to_csv(search_log_path, index=False)

    logging.info("Optimization completed.")
    logging.info(f"Optimized mapping values: {optimized_values}")
