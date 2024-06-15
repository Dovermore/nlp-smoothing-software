"""
CERTified Edit Distance defense (CERT-ED) authors authored this file

ChatGPT and/or Copilot are used in generating scaffolding code for this file
"""
import os
from typing import List, Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from utils import edit_distance_volume, l0_distance_volume


class CertifiedMetrics:
    def __init__(self, legends: List[str]):
        self.legends = legends
        self.reset()

    def reset(self):
        self.data = {"legend": [], "x": [], "y": [], "hue": [], "style": [], "size": []}

    @staticmethod
    def _certified_accuracy(
        preds: np.ndarray,
        labels: np.ndarray,
        certified_radii: np.ndarray,
        threshold: np.ndarray,
    ) -> float:
        return sum((preds == labels) & (certified_radii >= threshold)) / preds.shape[0]

    @staticmethod
    def _normalize_radii(
        all_certified_radii: List[np.ndarray], all_input_sizes: List[np.ndarray]
    ) -> List[np.ndarray]:
        return [
            (certified_radii / input_sizes) * 100
            for certified_radii, input_sizes in zip(
                all_certified_radii, all_input_sizes
            )
        ]

    def parse_certified_accuracy_data(
        self,
        all_preds: List[np.ndarray],
        all_labels: List[np.ndarray],
        all_certified_radii: List[np.ndarray],
        threshold_fr: float,
        threshold_to: float,
        hues: Optional[List[str]] = None,
        styles: Optional[List[str]] = None,
        sizes: Optional[List[str]] = None,
        log_scale: bool = False,
    ) -> pd.DataFrame:
        hues = hues if hues else [None] * len(self.legends)
        styles = styles if styles else [None] * len(self.legends)
        sizes = sizes if sizes else [None] * len(self.legends)

        thresholds = (
            np.geomspace(threshold_fr, threshold_to, 200)
            if log_scale
            else np.linspace(threshold_fr, threshold_to, 200)
        )

        for idx in range(len(self.legends)):
            for threshold in thresholds:
                ca = self._certified_accuracy(
                    all_preds[idx], all_labels[idx], all_certified_radii[idx], threshold
                )
                self.data["legend"].append(self.legends[idx])
                self.data["hue"].append(hues[idx])
                self.data["style"].append(styles[idx])
                self.data["size"].append(sizes[idx])
                self.data["x"].append(threshold)
                self.data["y"].append(ca)

        return pd.DataFrame(self.data)


def plot_certified_radius_accuracy(
    metrics: CertifiedMetrics,
    all_preds: List[np.ndarray],
    all_labels: List[np.ndarray],
    all_certified_radii: List[np.ndarray],
    all_input_sizes: Optional[List[np.ndarray]] = None,
    normalized: bool = False,
    threshold_fr: Optional[float] = None,
    threshold_to: Optional[float] = None,
    ax=None,
    show_acc: bool = False,
    upperbound: Optional[dict] = None,
    hues: Optional[List[str]] = None,
    sizes: Optional[List[str]] = None,
    styles: Optional[List[str]] = None,
    log_xscale: bool = False,
    y_lim=(0, 1),
    legend_kwargs: dict = {},
) -> (plt.Axes, pd.DataFrame):
    if normalized and all_input_sizes is None:
        raise ValueError("Provide file sizes to normalize the radii.")

    all_certified_radii = (
        metrics._normalize_radii(all_certified_radii, all_input_sizes)
        if normalized
        else all_certified_radii
    )
    threshold_fr = 1e-7 if log_xscale and threshold_fr is None else threshold_fr or 0
    threshold_to = threshold_to or max(np.max(cr) for cr in all_certified_radii) * 1.05

    data = metrics.parse_certified_accuracy_data(
        all_preds=all_preds,
        all_labels=all_labels,
        all_certified_radii=all_certified_radii,
        threshold_fr=threshold_fr,
        threshold_to=threshold_to,
        hues=hues,
        styles=styles,
        sizes=sizes,
        log_scale=log_xscale,
    )
    ax = sns.lineplot(
        data=data,
        x="x",
        y="y",
        hue="hue" if data["hue"].iloc[0] else None,
        style="style" if data["style"].iloc[0] else None,
        size="size" if data["size"].iloc[0] else None,
        sizes=(0.75, 1.25),
        ax=ax,
        legend=None,
    )

    ax.legend(handles=list(ax.get_lines()), labels=metrics.legends, **legend_kwargs)
    if show_acc:
        colors = []
        for child in ax.get_children():
            if isinstance(child, mpl.lines.Line2D):
                color = child.get_color()
                if len(child.get_data()[0]) > 0:
                    colors.append(color)

        # Annotate max y value
        ymaxs = data.groupby("legend")["y"].max()[metrics.legends]
        inc = 0
        diff = -0.02
        for legend, color in zip(ymaxs.index, colors):
            ymax = ymaxs[legend]
            ax.text(0, ymax + diff, f"{ymax:0.3f}", c=color)
            diff += inc

    if upperbound is not None:
        for line in upperbound["lines"]:
            ax.axvline(**line, **upperbound["kwargs"])

    # y axis style
    ax.yaxis.set_major_formatter(
        mpl.ticker.PercentFormatter(xmax=1.0, decimals=0, symbol="")
    )
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=0.1))
    ax.set_ylim(*y_lim)
    ax.set_ylabel("Certified accuracy ($\%$)")

    # x axis style
    if normalized:
        if not log_xscale:
            ax.xaxis.set_major_formatter(
                mpl.ticker.PercentFormatter(xmax=100, decimals=1, symbol="")
            )
        ax.set_xlabel("Normalized Radius, $r ($\%$)$")
    else:
        ax.set_xlabel("Radius, $r$")
    if log_xscale:
        ax.set_xscale("log")
    ax.set_xlim(threshold_fr, threshold_to)
    return ax, data


def plot_certified_volume_accuracy(
    metrics: CertifiedMetrics,
    all_preds: List[np.ndarray],
    all_labels: List[np.ndarray],
    all_certified_radii: List[np.ndarray],
    all_input_sizes: List[np.ndarray],
    all_vocab_size: List[int],
    all_threat_model: List[str],
    log_volume: Optional[bool] = False,
    threshold_fr: Optional[float] = None,
    threshold_to: Optional[float] = None,
    ax=None,
    show_acc: bool = False,
    upperbound: Optional[dict] = None,
    hues: Optional[List[str]] = None,
    sizes: Optional[List[str]] = None,
    styles: Optional[List[str]] = None,
    log_xscale: bool = False,
    y_lim=(0, 1),
    legend_kwargs: dict = {},
) -> (plt.Axes, pd.DataFrame):

    # Calculate the volume of the certified region
    all_certified_volumes = []
    for certified_radii, input_sizes, vocab_size, threat_model in zip(
        all_certified_radii, all_input_sizes, all_vocab_size, all_threat_model
    ):
        if threat_model == "l0":
            certified_volumes = np.array(
                [
                    l0_distance_volume(radius, input_size, vocab_size, log=log_volume)
                    for radius, input_size in zip(certified_radii, input_sizes)
                ]
            )
        elif threat_model == "edit":
            certified_volumes = np.array(
                [
                    edit_distance_volume(radius, input_size, vocab_size, log=log_volume)
                    for radius, input_size in zip(certified_radii, input_sizes)
                ]
            )
        else:
            raise ValueError(f"Unrecognized threat model: {threat_model}")
        all_certified_volumes.append(certified_volumes)

    threshold_fr = 1e-7 if log_xscale and threshold_fr is None else threshold_fr or 0
    threshold_to = (
        threshold_to or max(np.max(cr) for cr in all_certified_volumes) * 1.05
    )

    data = metrics.parse_certified_accuracy_data(
        all_preds=all_preds,
        all_labels=all_labels,
        all_certified_radii=all_certified_volumes,
        threshold_fr=threshold_fr,
        threshold_to=threshold_to,
        hues=hues,
        styles=styles,
        sizes=sizes,
        log_scale=log_xscale,
    )
    ax = sns.lineplot(
        data=data,
        x="x",
        y="y",
        hue="hue" if data["hue"].iloc[0] else None,
        style="style" if data["style"].iloc[0] else None,
        size="size" if data["size"].iloc[0] else None,
        sizes=(0.75, 1.25),
        ax=ax,
        legend=None,
    )

    ax.legend(handles=list(ax.get_lines()), labels=metrics.legends, **legend_kwargs)
    if show_acc:
        colors = []
        for child in ax.get_children():
            if isinstance(child, mpl.lines.Line2D):
                color = child.get_color()
                if len(child.get_data()[0]) > 0:
                    colors.append(color)

        # Annotate max y value
        ymaxs = data.groupby("legend")["y"].max()[metrics.legends]
        inc = 0
        diff = -0.02
        for legend, color in zip(ymaxs.index, colors):
            ymax = ymaxs[legend]
            ax.text(0, ymax + diff, f"{ymax:0.3f}", c=color)
            diff += inc

    if upperbound is not None:
        for line in upperbound["lines"]:
            ax.axvline(**line, **upperbound["kwargs"])

    # y axis style
    ax.yaxis.set_major_formatter(
        mpl.ticker.PercentFormatter(xmax=1.0, decimals=0, symbol="")
    )
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=0.1))
    ax.set_ylim(*y_lim)
    ax.set_ylabel("Certified accuracy ($\%$)")

    if log_volume:
        ax.set_xlabel("Log cardinality, $log(CC)$")
    else:
        ax.set_xlabel("Cardinality, $CC$")

    if log_xscale:
        ax.set_xscale("log")
    ax.set_xlim(threshold_fr, threshold_to)
    return ax, data


def plot_label_certified_radius_accuracy(
    metrics, labels, labels_kwargs, axs, *args, **kwargs
):
    zipped_label_kwargs = list(zip(labels, labels_kwargs, axs))
    labels_data = []
    for idx, (label, label_kwargs, ax) in enumerate(zipped_label_kwargs):
        metrics.reset()
        ax, data = plot_certified_radius_accuracy(
            *args, **label_kwargs, **kwargs, ax=ax, metrics=metrics,
        )
        # Remove xlabel except for the last one
        if idx < len(zipped_label_kwargs) - 1:
            ax.set_xlabel("")

        # Rmove legends except for the first one
        if idx > 0:
            ax.get_legend().remove()

        ax.set_ylabel(f"Certified {label} TPR ($\%$)")

        labels_data.append(data)
    return axs, labels_data


def plot_label_certified_volume_accuracy(
    metrics, labels, labels_kwargs, axs, *args, **kwargs
):
    zipped_label_kwargs = list(zip(labels, labels_kwargs, axs))
    labels_data = []
    for idx, (label, label_kwargs, ax) in enumerate(zipped_label_kwargs):
        metrics.reset()
        ax, data = plot_certified_volume_accuracy(
            *args, **label_kwargs, **kwargs, ax=ax, metrics=metrics,
        )
        # Remove xlabel except for the last one
        if idx < len(zipped_label_kwargs) - 1:
            ax.set_xlabel("")

        # Rmove legends except for the first one
        if idx > 0:
            ax.get_legend().remove()

        ax.set_ylabel(f"Certified {label} TPR ($\%$)")

        labels_data.append(data)
    return axs, labels_data


def training_history_plot(
    legends: List[str],
    tables: List[pd.DataFrame],
    y: str,
    ylabel: str,
    max_epochs=None,
    hues=None,
    styles=None,
    sizes=None,
    ax: plt.Axes = None,
    legend_kwargs: dict = dict(),
):
    if hues is None:
        hues = legends
    if styles is None:
        styles = legends
    for idx, legend in enumerate(legends):
        table = tables[idx]
        table["legend"] = legend
        table["hue"] = hues[idx] if hues else legend
        table["style"] = styles[idx] if styles else legend
        table["size"] = sizes[idx] if sizes else None
    data = pd.concat(tables, axis=0, ignore_index=True)
    x, xlabel = "epoch", "Epoch"
    if max_epochs is not None:
        data = data[data[x] < max_epochs]
    ax = sns.lineplot(data=data, x=x, y=y, style="style", hue="hue", ax=ax)
    ax.legend(handles=list(ax.get_lines()), labels=legends, **legend_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax, data


def process_certify_dirs(certify_dirs):
    all_labels = []
    all_preds = []
    all_certified_radii = []
    all_input_sizes = []

    for certify_dir in certify_dirs:
        table_path = os.path.join(certify_dir, "preds", "certified_predictions.csv")
        df = pd.read_csv(table_path)
        labels = df["label"]
        preds = df["pred"]
        certified_radii = np.where(
            df["pred"] == df["cr_pred"], df["certified_radius"], 0
        ).astype(np.int64)
        input_sizes = df["input"].str.split().str.len()
        certified_radii = np.minimum(certified_radii, input_sizes)

        all_labels.append(labels)
        all_preds.append(preds)
        all_certified_radii.append(certified_radii)
        all_input_sizes.append(input_sizes)

    return all_labels, all_preds, all_certified_radii, all_input_sizes


def group_data_by_label(all_labels, all_preds, all_certified_radii, all_input_sizes):
    unique_labels = sorted(np.unique(np.concatenate(all_labels)))
    out = {
        label: {
            "all_labels": [],
            "all_preds": [],
            "all_certified_radii": [],
            "all_input_sizes": [],
        }
        for label in unique_labels
    }
    for labels, preds, certified_radii, input_sizes in zip(
        all_labels, all_preds, all_certified_radii, all_input_sizes
    ):
        labels, preds, certified_radii, input_sizes = (
            np.array(labels),
            np.array(preds),
            np.array(certified_radii),
            np.array(input_sizes),
        )
        for label in unique_labels:
            idx = labels == label
            out[label]["all_labels"].append(labels[idx])
            out[label]["all_preds"].append(preds[idx])
            out[label]["all_certified_radii"].append(certified_radii[idx])
            out[label]["all_input_sizes"].append(input_sizes[idx])
    return unique_labels, out


def plot_figure(config):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["ps.usedistiller"] = "xpdf"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["figure.figsize"] = [3.34, 2.4]
    plt.rcParams["figure.dpi"] = 600

    # Load configs
    exp_name = config["exp_name"]
    output_dir = config["output_dir"]
    certify_dirs = config["certify_dirs"]
    legends = config["legends"]
    hues = config.get("hues", None)
    styles = config.get("styles", None)
    sizes = config.get("sizes", None)

    # plot_type = config["plot_type"].split("-", 1)[0]
    plot_type = config["plot_type"]
    plot_kwargs = config.get("plot_kwargs", dict())

    title = config["title"]
    fig_kwargs = config.get("fig_kwargs", dict())

    rcparams = config.get("rcparams", dict())
    for key in rcparams:
        plt.rcParams[key] = rcparams[key]

    if plot_type == "certified_radius_accuracy":
        all_labels, all_preds, all_certified_radii, all_input_sizes = (
            process_certify_dirs(certify_dirs)
        )
        for certify_dir in certify_dirs:
            table_path = os.path.join(certify_dir, "preds", "certified_predictions.csv")
            df = pd.read_csv(table_path)
            labels = df["label"]
            preds = df["pred"]
            certified_radii = np.where(
                df["pred"] == df["cr_pred"], df["certified_radius"], 0
            ).astype(np.int64)
            input_sizes = df["input"].str.split().str.len()
            certified_radii = np.minimum(certified_radii, input_sizes)

            all_labels.append(labels)
            all_preds.append(preds)
            all_certified_radii.append(certified_radii)
            all_input_sizes.append(input_sizes)

        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        cm = CertifiedMetrics(legends=legends)
        ax, data = plot_certified_radius_accuracy(
            metrics=cm,
            all_preds=all_preds,
            all_labels=all_labels,
            all_certified_radii=all_certified_radii,
            all_input_sizes=all_input_sizes,
            ax=ax,
            hues=hues,
            styles=styles,
            sizes=sizes,
            **plot_kwargs,
        )

    elif plot_type == "certified_volume_accuracy":
        all_labels, all_preds, all_certified_radii, all_input_sizes = (
            process_certify_dirs(certify_dirs)
        )

        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        cm = CertifiedMetrics(legends=legends)
        ax, data = plot_certified_volume_accuracy(
            metrics=cm,
            all_preds=all_preds,
            all_labels=all_labels,
            all_certified_radii=all_certified_radii,
            all_input_sizes=all_input_sizes,
            all_threat_model=config["threat_models"],
            all_vocab_size=config["vocab_sizes"],
            log_volume=config.get("log_volume", False),
            ax=ax,
            hues=hues,
            styles=styles,
            sizes=sizes,
            **plot_kwargs,
        )

    elif plot_type == "label_certified_radius_accuracy":
        all_labels, all_preds, all_certified_radii, all_input_sizes = (
            process_certify_dirs(certify_dirs)
        )
        label_ids, data = group_data_by_label(
            all_labels, all_preds, all_certified_radii, all_input_sizes
        )
        labels_kwargs = []
        for label in label_ids:
            label_kwargs = data[label]
            labels_kwargs.append(label_kwargs)

        fig, axs = plt.subplots(len(data), 1, **fig_kwargs)
        cm = CertifiedMetrics(legends=legends)
        axs, data = plot_label_certified_radius_accuracy(
            labels=config["labels"],
            labels_kwargs=labels_kwargs,
            axs=axs,
            metrics=cm,
            hues=hues,
            styles=styles,
            sizes=sizes,
            **plot_kwargs,
        )

    elif plot_type == "label_certified_volume_accuracy":
        all_labels, all_preds, all_certified_radii, all_input_sizes = (
            process_certify_dirs(certify_dirs)
        )
        label_ids, data = group_data_by_label(
            all_labels, all_preds, all_certified_radii, all_input_sizes
        )
        labels_kwargs = []
        for label in label_ids:
            label_kwargs = data[label]
            labels_kwargs.append(label_kwargs)

        fig, axs = plt.subplots(len(data), 1, **fig_kwargs)
        cm = CertifiedMetrics(legends=legends)
        axs, data = plot_label_certified_volume_accuracy(
            labels=config["labels"],
            labels_kwargs=labels_kwargs,
            all_threat_model=config["threat_models"],
            all_vocab_size=config["vocab_sizes"],
            log_volume=config.get("log_volume", False),
            axs=axs,
            metrics=cm,
            hues=hues,
            styles=styles,
            sizes=sizes,
            **plot_kwargs,
        )

    else:
        raise ValueError(f"Unrecongnized plot_type: {plot_type}")
    fig.suptitle(title)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, exp_name + ".pdf"), format="pdf", dpi=600)
    fig.savefig(os.path.join(output_dir, exp_name + ".png"), format="png", dpi=600)
    fig.savefig(os.path.join(output_dir, exp_name + ".svg"), format="svg", dpi=600)
    plt.close(fig)
