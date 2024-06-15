"""
CERTified Edit Distance defense (CERT-ED) authors authored this file

ChatGPT and/or Copilot are used in generating scaffolding code for this file
"""
import os
import logging
import json

import torch
from textattack.attacker import AttackArgs, Attacker
from textattack.datasets import HuggingFaceDataset

import adv_attack
from adv_attack import SmoothedClassifierAttackWrapper
from utils import load_components, subsample_dataset


def attack_model(config):
    log_dir = config["log_dir"]
    output_dir = config["output_dir"]
    
    if "dataset" in config:
        config["model_config"] = config["dataset"]

    components = load_components(config["model_config"])
    smoothed_model = components["smoothed_model"]
    model_wrapper = SmoothedClassifierAttackWrapper(
        smoothed_model,
        **config["model_wrapper_args"],
    )
    dataset = components["test_dataset"].data
    if config["shuffle"]:
        indices, dataset = subsample_dataset(
            dataset=dataset, sample_size=config["num_examples"], seed=config["seed"]
        )
    else:
        indices = list(range(len(dataset)))[: config["num_examples"]]
        dataset = dataset.select(indices)
    indices_json = os.path.join(output_dir, f"indices.csv")
    with open(indices_json, "w") as f:
        json.dump(indices, f)
    dataset = HuggingFaceDataset(dataset)
    attack_class = getattr(adv_attack, config["attack"])
    attack = attack_class.build(model_wrapper=model_wrapper)
    attack.time_budget = config["time_budget"]

    for i in range(config["num_attacks"]):
        separator = "-" * 50
        separator_line = f"{separator} Attack: {i:02d} {separator}"
        logging.info(separator_line)
        logging.error(separator_line)

        checkpoint_dir = os.path.join(config["checkpoint_dir"], f"{i:02d}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_csv = os.path.join(log_dir, f"{i:02d}.csv")
        log_json = os.path.join(log_dir, f"{i:02d}.json")
        log_txt = os.path.join(log_dir, f"{i:02d}.txt")

        checkpoints = [
            os.path.join(checkpoint_dir, fname)
            for fname in os.listdir(checkpoint_dir)
            if fname.endswith(".chkpt")
        ]
        if config["load_checkpoint"] and len(checkpoints) > 0:
            timestamps = [
                int(checkpoint.split("/")[-1].split(".")[0])
                for checkpoint in checkpoints
            ]
            last_idx = timestamps.index(max(timestamps))
            print(f"Recovering checkpoint: {checkpoints[last_idx]}")
            attacker = Attacker.from_checkpoint(attack, dataset, checkpoints[last_idx])
        else:
            attacker = Attacker(
                attack,
                dataset,
                attack_args=AttackArgs(
                    num_examples=-1,  # entire dataset
                    checkpoint_interval=config["checkpoint_interval"],
                    checkpoint_dir=checkpoint_dir,
                    log_to_csv=log_csv,
                    log_to_txt=log_txt,
                    log_summary_to_json=log_json,
                    num_workers_per_device=config["num_workers_per_device"],
                    parallel=True if torch.cuda.device_count() > 1 else False,
                    enable_advance_metrics=True,
                    random_seed=config["seed"] + i,
                    query_budget=config["query_budget"],
                    disable_stdout=True,
                ),
            )
        results = attacker.attack_dataset()
        for result in results:
            logging.info(result.__str__(color_method="ansi"))
