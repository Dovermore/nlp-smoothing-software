"""
CERTified Edit Distance defense (CERT-ED) authors authored this file

ChatGPT and/or Copilot are used in generating scaffolding code for this file
"""
import logging
import os

import torch
import yaml


TRAIN_DEFAULTS = {
    "use_gpu": True,
    "model": {"type": "roberta-base"},
    "optimizer": "AdamW",
    "optimizer_args": {"lr": 2e-5},
    "scheduler": "linear_schedule_with_warmup",
    "scheduler_args": {},
    "grad_clip": 1,
    "num_workers": torch.get_num_threads(),
    "batch_size": 32,
    "early_stopping_patience": 5,
    "num_samples": 32,
    "evaluate_epoch": 5,
    "keep_checkpoints": 3,
    "max_epochs": 200,
    "update_step": 100,
    "seed": 42,
    "valid_size": 0.1,
    "sample_size": None,
    "mask_grad_scale": None,
}

CERTIFY_DEFAULTS = {
    "use_gpu": True,
    "batch_size": 32,
    "pred_num_samples": 100,
    "pred_kwargs": {},
    "cr_num_samples": 1000, 
    "cr_kwargs": {"alpha": 0.05},
    "seed": 42,
    "test_size": None,  # use the entire test set by default (None)
    "checkpoint_interval": 100,
    "load_df": False,
}

ATTACK_DEFAULTS = {
    "seed": 42,
    "shuffle": True,
    "num_examples": 100,
    "checkpoint_interval": 10,
    "load_checkpoint": True,
    "num_attacks": 1,
    "query_budget": None,
    "time_budget": 600,
    "num_workers_per_device": 2,
}


def set_defaults(config, default_values):
    # Define default values for each key
    for key, default_value in default_values.items():
        if key not in config:
            config[key] = default_value
        elif isinstance(config[key], dict):
            for subkey, sub_default_value in default_values[key].items():
                if subkey not in config[key]:
                    config[key][subkey] = sub_default_value
    return config


def save_yaml(data, path, safe=True):
    """Utility function to save a dictionary to a YAML file."""
    if safe and os.path.exists(path):
        with open(path, "r") as f:
            other_data = yaml.safe_load(f)
        if data != other_data:
            raise ValueError("there is already one config file in the output directory and it is different from the current one.")
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def load_config(config_path, mode, save=False, safe=True):
    """General function to load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["exp_name"] = config.get(
        "exp_name", os.path.splitext(os.path.basename(config_path))[0]
    )
    # Use the output_root and exp_name to create a unique output directory if output_dir is not specified
    if config.get("output_dir") is None:
        config["output_dir"] = os.path.join(
            config["output_root"], f"{config['exp_name']}"
        )
    config["stdout_log"] = os.path.join(config["output_dir"], "stdout.log")
    config["stderr_log"] = os.path.join(config["output_dir"], "stderr.log")
    # Post-loading processing callback
    if mode == "train":
        post_train_load(config)
    elif mode == "certify":
        post_certify_load(config)
    elif mode == "plot":
        post_plot_load(config)
    elif mode == "attack":
        post_attack_load(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # Save the config to the output directory
    save_path = os.path.join(config["output_dir"], "config.yaml")
    if save:
        os.makedirs(config["output_dir"], exist_ok=True)
        save_yaml(config, save_path, safe=safe)
    logging.info(f"Configuration loaded from {config_path}.")
    return config


def post_plot_load(config):
    """Post-load actions specific to training configuration."""
    return config


def post_train_load(config):
    """Post-load actions specific to training configuration."""
    config["pred_dir"] = os.path.join(config["output_dir"], "preds")
    config["checkpoint_dir"] = os.path.join(config["output_dir"], "checkpoints")
    config["log_dir"] = os.path.join(config["output_dir"], "logs")
    set_defaults(config, TRAIN_DEFAULTS)


def post_certify_load(config, load_checkpoint="best"):
    """Post-load actions specific to certification configuration."""
    # Where results should be saved
    config["pred_dir"] = os.path.join(config["output_dir"], "preds")
    # Optionally save/read repeat_forward
    config["log_dir"] = os.path.join(config["output_dir"], "logs")
    # Set default and load model
    set_defaults(config, CERTIFY_DEFAULTS)
    model_config_path = os.path.join(config["model_dir"], "config.yaml")
    config["model_config"] = load_config(model_config_path, mode="train", save=False)
    config["model_config"]["load_checkpoint"] = load_checkpoint
    return config


def post_attack_load(config, load_checkpoint="best"):
    # Where results should be saved
    config["pred_dir"] = os.path.join(config["output_dir"], "preds")
    config["checkpoint_dir"] = os.path.join(config["output_dir"], "checkpoints")
    config["log_dir"] = os.path.join(config["output_dir"], "logs")
    # Set default and load model
    set_defaults(config, ATTACK_DEFAULTS)
    model_config_path = os.path.join(config["model_dir"], "config.yaml")
    config["model_config"] = load_config(model_config_path, mode="train", save=False)
    config["model_config"]["load_checkpoint"] = load_checkpoint
    return config
