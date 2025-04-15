"""
ChatGPT and/or Copilot are used in generating scaffolding code for this file
"""
import os
from typing import List, Optional

os.environ["DISABLE_TF"] = "1"

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


def align_axs_ranges(axs):
    """
    Align the x and y axis limits of a list of Matplotlib axes to the last one.

    Args:
        axs (list): List of Matplotlib axes objects.
    """
    if not axs:
        return

    # Get limits of the last axis in the list
    xlim_last = axs[-1].get_xlim()
    ylim_last = axs[-1].get_ylim()

    # Apply the limits to all axes
    for ax in axs[:-1]:
        ax.set_xlim(xlim_last)
        ax.set_ylim(ylim_last)
        ax.xaxis.set_ticklabels([])


def configure_legend(ax, legends, legend_kwargs, transpose=False):
    """
    Configure the legend for the provided axis, with an optional transpose arrangement.

    Parameters:
        ax (matplotlib.axes.Axes): The axis object.
        legends (list): List of legend labels.
        legend_kwargs (dict): Additional keyword arguments for `ax.legend`.
        transpose (bool): Whether to transpose the legend arrangement.
    """
    handles = list(ax.get_lines())

    if transpose:
        # Reorganize handles and labels to create row-wise legend arrangement
        nrow = legend_kwargs.get("ncol", 1)  # Row should be previous col
        ncol = (len(handles) + 1) // nrow  # Col should be prevoius row
        legend_kwargs["ncol"] = ncol
        reordered_handles, reordered_labels = [], []

        for col in range(ncol):
            for row in range(nrow):
                index = col + row * ncol  # Transpose index calculation
                if index < len(handles):
                    reordered_handles.append(handles[index])
                    reordered_labels.append(legends[index])

        handles, legends = reordered_handles, reordered_labels

    # Apply legend to the axis
    ax.legend(handles=handles, labels=legends, **legend_kwargs)


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
) -> tuple[plt.Axes, pd.DataFrame]:
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

    if "transpose" in legend_kwargs:
        transpose = legend_kwargs["transpose"]
        del legend_kwargs["transpose"]
    else:
        transpose = False

    configure_legend(ax, metrics.legends, legend_kwargs, transpose=transpose)
    # ax.legend(handles=list(ax.get_lines()), labels=metrics.legends, **legend_kwargs)
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
) -> tuple[plt.Axes, pd.DataFrame]:
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

    if "transpose" in legend_kwargs:
        transpose = legend_kwargs["transpose"]
        del legend_kwargs["transpose"]
    else:
        transpose = False

    configure_legend(ax, metrics.legends, legend_kwargs, transpose=transpose)
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
            *args,
            **label_kwargs,
            **kwargs,
            ax=ax,
            metrics=metrics,
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
            *args,
            **label_kwargs,
            **kwargs,
            ax=ax,
            metrics=metrics,
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


def plot_quantile_certified_radius_accuracy(
    metrics, quantile_ranges, quantile_kwargs, axs, *args, **kwargs
):
    """
    Plot certified radius accuracy grouped by quantiles of input sizes.

    Parameters:
        metrics (CertifiedMetrics): Metrics object to track data.
        quantile_ranges (list of tuples): List of quantile ranges.
        quantile_kwargs (list): List of data dictionaries for each quantile.
        axs (list of Axes): Subplots axes.
        *args: Additional arguments for `plot_certified_radius_accuracy`.
        **kwargs: Additional keyword arguments for `plot_certified_radius_accuracy`.

    Returns:
        axs: Updated subplots axes.
        quantiles_data: List of data for each quantile range.
    """
    zipped_quantile_kwargs = list(zip(quantile_ranges, quantile_kwargs, axs))
    quantiles_data = []
    for idx, (quantile_range, quantile_kwargs, ax) in enumerate(zipped_quantile_kwargs):
        metrics.reset()
        ax, data = plot_certified_radius_accuracy(
            *args,
            **quantile_kwargs,
            **kwargs,
            ax=ax,
            metrics=metrics,
        )
        # Remove xlabel except for the last one
        if idx < len(zipped_quantile_kwargs) - 1:
            ax.set_xlabel("")

        # Remove legends except for the first one
        if idx > 0:
            ax.get_legend().remove()

        # Set ylabel to reflect quantile range
        ax.set_ylabel(f"CertAcc ($\%$)")

        quantiles_data.append(data)
    return axs, quantiles_data


def plot_quantile_certified_volume_accuracy(
    metrics, quantile_ranges, quantile_kwargs, axs, *args, **kwargs
):
    """
    Plot certified volume accuracy grouped by quantiles of input sizes.

    Parameters:
        metrics (CertifiedMetrics): Metrics object to track data.
        quantile_ranges (list of tuples): List of quantile ranges.
        quantile_kwargs (list): List of data dictionaries for each quantile.
        axs (list of Axes): Subplots axes.
        *args: Additional arguments for `plot_certified_volume_accuracy`.
        **kwargs: Additional keyword arguments for `plot_certified_volume_accuracy`.

    Returns:
        axs: Updated subplots axes.
        quantiles_data: List of data for each quantile range.
    """
    zipped_quantile_kwargs = list(zip(quantile_ranges, quantile_kwargs, axs))
    quantiles_data = []
    for idx, (quantile_range, quantile_kwargs, ax) in enumerate(zipped_quantile_kwargs):
        metrics.reset()
        ax, data = plot_certified_volume_accuracy(
            *args,
            **quantile_kwargs,
            **kwargs,
            ax=ax,
            metrics=metrics,
        )
        # Remove xlabel except for the last one
        if idx < len(zipped_quantile_kwargs) - 1:
            ax.set_xlabel("")

        # Remove legends except for the first one
        if idx > 0:
            ax.get_legend().remove()

        # Set ylabel to reflect quantile range
        ax.set_ylabel(f"CertAcc ($\%$)")

        quantiles_data.append(data)
    return axs, quantiles_data


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

    if "transpose" in legend_kwargs:
        transpose = legend_kwargs["transpose"]
        del legend_kwargs["transpose"]
    else:
        transpose = False

    configure_legend(ax, legends, legend_kwargs, transpose=transpose)
    # ax.legend(handles=list(ax.get_lines()), labels=legends, **legend_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax, data


def process_certify_dirs(certify_dirs, remove_outliers=None):
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
        input_sizes = df["input"].str.split(" ").str.len()
        certified_radii = np.minimum(certified_radii, input_sizes)

        if remove_outliers is not None:
            # Filter top remove_outliers percent of certified_radii
            threshold = np.percentile(certified_radii, 100 - remove_outliers)
            idx = certified_radii <= threshold
            labels, preds, certified_radii, input_sizes = (
                labels[idx],
                preds[idx],
                certified_radii[idx],
                input_sizes[idx],
            )

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


def group_data_by_quantile(
    all_labels, all_preds, all_certified_radii, all_input_sizes, quantiles
):
    """
    Group data by quantile of input_sizes.

    Parameters:
        all_labels (list of arrays): List of label arrays for each dataset.
        all_preds (list of arrays): List of prediction arrays for each dataset.
        all_certified_radii (list of arrays): List of certified radius arrays for each dataset.
        all_input_sizes (list of arrays): List of input size arrays for each dataset.
        quantiles (list): List of quantile thresholds (e.g., [0.25, 0.5, 0.75]).

    Returns:
        quantile_ranges (list of tuples): List of quantile range tuples (e.g., [(min, q1), (q1, q2), ...]).
        grouped_data (dict): Dictionary of grouped data by quantile range.
    """
    all_input_sizes_combined = np.concatenate(all_input_sizes)
    quantile_values = np.quantile(all_input_sizes_combined, quantiles)

    # Define quantile ranges
    quantile_ranges = []
    previous = 0
    for q in quantile_values:
        quantile_ranges.append((previous, q))
        previous = q
    quantile_ranges.append((previous, float("inf")))

    grouped_data = {
        range_idx: {
            "all_labels": [],
            "all_preds": [],
            "all_certified_radii": [],
            "all_input_sizes": [],
        }
        for range_idx in range(len(quantile_ranges))
    }

    # Group data by quantile
    for labels, preds, certified_radii, input_sizes in zip(
        all_labels, all_preds, all_certified_radii, all_input_sizes
    ):
        labels, preds, certified_radii, input_sizes = (
            np.array(labels),
            np.array(preds),
            np.array(certified_radii),
            np.array(input_sizes),
        )
        for range_idx, (lower, upper) in enumerate(quantile_ranges):
            idx = (input_sizes > lower) & (input_sizes <= upper)
            grouped_data[range_idx]["all_labels"].append(labels[idx])
            grouped_data[range_idx]["all_preds"].append(preds[idx])
            grouped_data[range_idx]["all_certified_radii"].append(certified_radii[idx])
            grouped_data[range_idx]["all_input_sizes"].append(input_sizes[idx])

    return quantile_ranges, grouped_data


# Generic plot handler
def handle_plot(plot_function, **kwargs):
    fig, ax_or_axs = plot_function(**kwargs)
    return fig, ax_or_axs


# Certified Radius Accuracy plot
def plot_certified_radius_accuracy_handler(
    all_preds, all_labels, all_certified_radii, all_input_sizes, **kwargs
):
    fig, ax = plt.subplots(1, 1, **kwargs.get("fig_kwargs", {}))
    metrics = CertifiedMetrics(legends=kwargs.get("legends"))
    ax, data = plot_certified_radius_accuracy(
        metrics=metrics,
        all_preds=all_preds,
        all_labels=all_labels,
        all_certified_radii=all_certified_radii,
        all_input_sizes=all_input_sizes,
        ax=ax,
        hues=kwargs.get("hues"),
        styles=kwargs.get("styles"),
        sizes=kwargs.get("sizes"),
        **kwargs.get("plot_kwargs", {}),
    )
    return fig, ax


# Certified Volume Accuracy plot
def plot_certified_volume_accuracy_handler(
    all_preds, all_labels, all_certified_radii, all_input_sizes, **kwargs
):
    fig, ax = plt.subplots(1, 1, **kwargs.get("fig_kwargs", {}))
    metrics = CertifiedMetrics(legends=kwargs.get("legends"))
    ax, data = plot_certified_volume_accuracy(
        metrics=metrics,
        all_preds=all_preds,
        all_labels=all_labels,
        all_certified_radii=all_certified_radii,
        all_input_sizes=all_input_sizes,
        all_threat_model=kwargs.get("threat_models"),
        all_vocab_size=kwargs.get("vocab_sizes"),
        log_volume=kwargs.get("log_volume", False),
        ax=ax,
        hues=kwargs.get("hues"),
        styles=kwargs.get("styles"),
        sizes=kwargs.get("sizes"),
        **kwargs.get("plot_kwargs", {}),
    )
    return fig, ax


# Label-specific Certified Radius Accuracy plot
def plot_label_certified_radius_accuracy_handler(labels_kwargs, **kwargs):
    fig, axs = plt.subplots(len(labels_kwargs), 1, **kwargs.get("fig_kwargs", {}))
    metrics = CertifiedMetrics(legends=kwargs.get("legends"))
    axs, data = plot_label_certified_radius_accuracy(
        labels=kwargs.get("labels"),
        labels_kwargs=labels_kwargs,
        axs=axs,
        metrics=metrics,
        hues=kwargs.get("hues"),
        styles=kwargs.get("styles"),
        sizes=kwargs.get("sizes"),
        **kwargs.get("plot_kwargs", {}),
    )
    align_axs = kwargs.get("align_axs", False)
    if align_axs:
        align_axs_ranges(axs)
    return fig, axs


# Label-specific Certified Volume Accuracy plot
def plot_label_certified_volume_accuracy_handler(labels_kwargs, **kwargs):
    fig, axs = plt.subplots(len(labels_kwargs), 1, **kwargs.get("fig_kwargs", {}))
    metrics = CertifiedMetrics(legends=kwargs.get("legends"))
    axs, data = plot_label_certified_volume_accuracy(
        labels=kwargs.get("labels"),
        labels_kwargs=labels_kwargs,
        all_threat_model=kwargs.get("threat_models"),
        all_vocab_size=kwargs.get("vocab_sizes"),
        log_volume=kwargs.get("log_volume", False),
        axs=axs,
        metrics=metrics,
        hues=kwargs.get("hues"),
        styles=kwargs.get("styles"),
        sizes=kwargs.get("sizes"),
        **kwargs.get("plot_kwargs", {}),
    )
    align_axs = kwargs.get("align_axs", False)
    if align_axs:
        align_axs_ranges(axs)
    return fig, axs


# Quantile Certified Radius Accuracy plot
def plot_quantile_certified_radius_accuracy_handler(
    quantile_ranges, quantile_kwargs, all_input_sizes, **kwargs
):
    include_inverse_cdf = kwargs.get("include_inverse_cdf", True)
    gridspec_kwargs = kwargs.get(
        "gridspec_kwargs",
        {
            "width_ratios": [2, 1],
            "hspace": 0.2,
            "wspace": 0.1,
        },
    )
    fig, left_axes, right_axis = setup_figure_with_axes(
        len(quantile_ranges),
        kwargs.get("fig_kwargs", {}),
        include_inverse_cdf,
        gridspec_kwargs,
        all_input_sizes[0],
    )
    metrics = CertifiedMetrics(legends=kwargs.get("legends"))
    left_axes, data = plot_quantile_certified_radius_accuracy(
        quantile_ranges=quantile_ranges,
        quantile_kwargs=quantile_kwargs,
        axs=left_axes,
        metrics=metrics,
        hues=kwargs.get("hues"),
        styles=kwargs.get("styles"),
        sizes=kwargs.get("sizes"),
        **kwargs.get("plot_kwargs", {}),
    )
    align_axs = kwargs.get("align_axs", False)
    if align_axs:
        align_axs_ranges(left_axes)
    return fig, left_axes


# Quantile Certified Volume Accuracy plot
def plot_quantile_certified_volume_accuracy_handler(
    quantile_ranges, quantile_kwargs, all_input_sizes, **kwargs
):
    include_inverse_cdf = kwargs.get("include_inverse_cdf", True)
    gridspec_kwargs = kwargs.get(
        "gridspec_kwargs",
        {
            "width_ratios": [2, 1],
            "hspace": 0.2,
            "wspace": 0.1,
        },
    )
    fig, left_axes, right_axis = setup_figure_with_axes(
        len(quantile_ranges),
        kwargs.get("fig_kwargs", {}),
        include_inverse_cdf,
        gridspec_kwargs,
        all_input_sizes[0],
    )
    metrics = CertifiedMetrics(legends=kwargs.get("legends"))
    left_axes, data = plot_quantile_certified_volume_accuracy(
        quantile_ranges=quantile_ranges,
        quantile_kwargs=quantile_kwargs,
        axs=left_axes,
        metrics=metrics,
        hues=kwargs.get("hues"),
        styles=kwargs.get("styles"),
        sizes=kwargs.get("sizes"),
        log_volume=kwargs.get("log_volume", False),
        all_threat_model=kwargs.get("threat_models"),
        all_vocab_size=kwargs.get("vocab_sizes"),
        **kwargs.get("plot_kwargs", {}),
    )
    align_axs = kwargs.get("align_axs", False)
    if align_axs:
        align_axs_ranges(left_axes)

    return fig, left_axes


# Inverse CDF setup
def setup_inverse_cdf_axes(fig, num_rows, gridspec_kwargs, input_sizes, invert=False):
    grid = plt.GridSpec(
        num_rows,
        2,
        width_ratios=gridspec_kwargs["width_ratios"],
        hspace=gridspec_kwargs["hspace"],
        wspace=gridspec_kwargs["wspace"],
    )
    left_axes = [fig.add_subplot(grid[i, 0]) for i in range(num_rows)]
    right_axis = fig.add_subplot(grid[:, 1])

    # Plot inverse CDF on the right axis
    input_sizes = np.array(input_sizes)
    lower_bound = np.percentile(input_sizes, 5)
    upper_bound = np.percentile(input_sizes, 100 - 5)
    input_sizes = input_sizes[
        (input_sizes >= lower_bound) & (input_sizes <= upper_bound)
    ]

    sorted_sizes = np.sort(input_sizes)
    cdf_values = np.linspace(0, 1, len(sorted_sizes))
    if invert:
        right_axis.plot(cdf_values, sorted_sizes)
        right_axis.set_ylabel("Input Size")
        right_axis.yaxis.tick_right()
        right_axis.yaxis.set_label_position("right")
        right_axis.set_xlabel("Quantile")

        right_axis.set_xlim([0, 1])
        right_axis.set_ylim([sorted_sizes[0], sorted_sizes[-1]])
        right_axis.invert_yaxis()

        # Add bars at 1/num_rows intervals
        num_bars = num_rows - 1
        bar_positions = np.linspace(0, 1, num_bars + 2)
        for bar in bar_positions[1:-1]:
            right_axis.axvline(bar, color="k", linestyle="--", linewidth=1)

        # Align ticks with the bars
        right_axis.set_xticks(bar_positions)
        right_axis.set_xticklabels([f"{bar:.2f}" for bar in bar_positions])
    else:
        right_axis.plot(sorted_sizes, cdf_values)
        right_axis.set_xlabel("Input Size")
        right_axis.yaxis.tick_right()
        right_axis.yaxis.set_label_position("right")
        right_axis.set_ylim([0, 1])
        right_axis.invert_yaxis()
        right_axis.set_ylabel("Quantile")

        # Add bars at 1/num_rows intervals
        num_bars = num_rows - 1
        bar_positions = np.linspace(0, 1, num_bars + 2)
        for bar in bar_positions[1:-1]:
            right_axis.axhline(bar, color="k", linestyle="--", linewidth=1)

        # Align ticks with the bars
        right_axis.set_yticks(bar_positions)
        right_axis.set_yticklabels([f"{bar:.2f}" for bar in bar_positions])

        ## Adjust x ticks to match quantile positions including start and end
        #right_axis.set_xticks([sorted_sizes[0]] + list(sorted_sizes[::len(sorted_sizes) // num_rows]) + [sorted_sizes[-1]])
        #right_axis.set_xticklabels([f"{x:.2f}" for x in ([sorted_sizes[0]] + list(sorted_sizes[::len(sorted_sizes) // num_rows]) + [sorted_sizes[-1]])])

        right_axis.set_xlim([sorted_sizes[0], sorted_sizes[-1]])
        right_axis.set_xticks([sorted_sizes[0], sorted_sizes[-1]])

    return left_axes, right_axis


# Figure setup with optional inverse CDF
def setup_figure_with_axes(
    num_axes, fig_kwargs, include_inverse_cdf, gridspec_kwargs, input_sizes
):
    if include_inverse_cdf:
        fig = plt.figure(**fig_kwargs)
        left_axes, right_axis = setup_inverse_cdf_axes(
            fig, num_axes, gridspec_kwargs, input_sizes
        )
        return fig, left_axes, right_axis
    else:
        fig, axes = plt.subplots(num_axes, 1, **fig_kwargs)
        return fig, axes, None


# Main plot_figure function
def plot_figure(config):
    plt.rcParams.update(
        {
            "text.usetex": True,
            "ps.usedistiller": "xpdf",
            "figure.facecolor": "white",
            "figure.figsize": [3.34, 2.4],
            "figure.dpi": 600,
        }
    )

    for key, value in config.get("rcparams", {}).items():
        plt.rcParams[key] = value

    plot_type = config["plot_type"]
    remove_outliers = config.pop("remove_outliers", 0.01)
    all_labels, all_preds, all_certified_radii, all_input_sizes = process_certify_dirs(
        config["certify_dirs"], remove_outliers=remove_outliers
    )

    handlers = {
        "certified_radius_accuracy": plot_certified_radius_accuracy_handler,
        "certified_volume_accuracy": plot_certified_volume_accuracy_handler,
        "label_certified_radius_accuracy": plot_label_certified_radius_accuracy_handler,
        "label_certified_volume_accuracy": plot_label_certified_volume_accuracy_handler,
        "quantile_certified_radius_accuracy": plot_quantile_certified_radius_accuracy_handler,
        "quantile_certified_volume_accuracy": plot_quantile_certified_volume_accuracy_handler,
    }

    if plot_type in handlers:
        if "quantile" in plot_type:
            quantile_ranges, data = group_data_by_quantile(
                all_labels,
                all_preds,
                all_certified_radii,
                all_input_sizes,
                config["quantiles"],
            )
            quantile_kwargs = [data[q_idx] for q_idx in range(len(quantile_ranges))]
            fig, ax_or_axs = handle_plot(
                handlers[plot_type],
                quantile_ranges=quantile_ranges,
                quantile_kwargs=quantile_kwargs,
                all_input_sizes=all_input_sizes,
                **config,
            )
        elif "label" in plot_type:
            label_ids, data = group_data_by_label(
                all_labels, all_preds, all_certified_radii, all_input_sizes
            )
            labels_kwargs = [data[label] for label in label_ids]
            fig, ax_or_axs = handle_plot(
                handlers[plot_type], labels_kwargs=labels_kwargs, **config
            )
        else:
            fig, ax_or_axs = handle_plot(
                handlers[plot_type],
                all_preds=all_preds,
                all_labels=all_labels,
                all_certified_radii=all_certified_radii,
                all_input_sizes=all_input_sizes,
                **config,
            )
    else:
        raise ValueError(f"Unrecognized plot_type: {plot_type}")

    fig.suptitle(config["title"])
    os.makedirs(config["output_dir"], exist_ok=True)
    fig.savefig(
        os.path.join(config["output_dir"], config["exp_name"] + ".pdf"),
        format="pdf",
        dpi=600,
    )
    fig.savefig(
        os.path.join(config["output_dir"], config["exp_name"] + ".png"),
        format="png",
        dpi=600,
    )
    fig.savefig(
        os.path.join(config["output_dir"], config["exp_name"] + ".svg"),
        format="svg",
        dpi=600,
    )
    plt.close(fig)
