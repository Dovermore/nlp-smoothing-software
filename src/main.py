"""
CERTified Edit Distance defense (CERT-ED) authors authored this file

ChatGPT and/or Copilot are used in generating scaffolding code for this file
"""
import argparse
import logging
import warnings

from certify import certify_model
from train import train_model
from visualization import plot_figure
from config_loader import load_config
from attack import attack_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep Learning Training and Certification"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "certify", "plot", "attack"],
        required=True,
        help="Whether to train or certify the model",
    )
    parser.add_argument(
        "--config_path",
        required=True,
        help="Where the config file is for this experiment",
    )
    parser.add_argument(
        "--override_config",
        action="store_true",
        default=False,
        help="If set, override existing config in the output directory",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level for the application",
    )
    parser.add_argument(
        "--ignore_transformers_warnings",
        action="store_true",
        default=True,
        help="If set, will ignore warnings from the transformers module",
    )
    parser.add_argument(
        "--ignore_seaborn_warnings",
        action="store_true",
        default=True,
        help="If set, will ignore future warnings seaborn module",
    )
    args = parser.parse_args()
    return args

def redirect_logging(config, log_level):
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stdout_handler = logging.FileHandler(config["stdout_log"], mode="a")
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    stderr_handler = logging.FileHandler(config["stderr_log"], mode="a")
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)


def main():
    args = parse_args()
    if args.ignore_transformers_warnings:
        import transformers
        import datasets
        transformers.logging.set_verbosity_error()
        datasets.logging.set_verbosity_error()

    if args.ignore_seaborn_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn._oldcore')

    config = load_config(config_path=args.config_path, mode=args.mode, save=True, safe=not args.override_config)
    redirect_logging(config, args.log_level)

    # Print line separator in both log files
    separator = "=" * 50
    separator_line = f"{separator} Mode: {args.mode} {separator}"
    logging.info(separator_line)
    logging.error(separator_line)

    if args.mode == "train":
        train_model(config)
    elif args.mode == "certify":
        certify_model(config)
    elif args.mode == "plot":
        plot_figure(config)
    elif args.mode == "attack":
        attack_model(config)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
