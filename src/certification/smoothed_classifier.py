"""
CERTified Edit Distance defense (CERT-ED) authors edited this file

Some codes are from the RS-Del code repository
"""
from dataclasses import asdict, dataclass
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from . import PerturbationTokenizer
from .utils import RepeatSampleDataset


@dataclass
class CertifiedPrediction:
    pred: int = None
    pred_label_counts: np.ndarray = None
    pred_pval: float = None

    certified_radius: float = None
    cr_label_counts: np.ndarray = None
    cr_pred: int = None

    def __post_init__(self):
        pass

    def __add__(self, other):
        return CertifiedPrediction(
            pred=self.pred if self.pred is not None else other.pred,
            pred_label_counts=self.pred_label_counts
            if self.pred_label_counts is not None
            else other.pred_label_counts,
            pred_pval=self.pred_pval if self.pred_pval is not None else other.pred_pval,
            certified_radius=self.certified_radius
            if self.certified_radius is not None
            else other.certified_radius,
            cr_label_counts=self.cr_label_counts
            if self.cr_label_counts is not None
            else other.cr_label_counts,
            cr_pred=self.cr_pred if self.cr_pred is not None else other.cr_pred,
        )


class SmoothedClassifierWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        perturbation_tokenizer: PerturbationTokenizer,
        max_length: int = 256,
        device: Union[torch.device, str, None] = None,
    ):
        self.model = model
        self.perturbation_tokenizer = perturbation_tokenizer
        self.max_length = max_length
        self.device = device if device else next(iter(model.parameters())).device
        self.model.to(self.device)

    def agg_repeat_forward(
        self,
        text: str,
        num_samples: int,
        aggregation: str = "hard",
        batch_size: int = 32,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Perform either hard or soft aggregation on the classifier's preds.

        Args:
        - text: The input text string to be classified.
        - num_samples: Number of Monte Carlo samples to use.
        - aggregation: Either 'hard' or 'soft'. 'hard' will count the predicted labels, 'soft' will average the logits.
        - batch_size: Number of samples to pass to the classifier in one call.

        Returns:
        - Tensor with shape (num_labels,) representing either label counts or averaged logits.
        """
        assert aggregation in [
            "hard",
            "soft",
            "logit",
        ], "Aggregation must be either 'hard', 'soft' or 'logit'"

        training = self.training
        self.train(False)

        dataset = RepeatSampleDataset(
            text,
            self.perturbation_tokenizer,
            num_samples=num_samples,
            max_length=self.max_length,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(self.perturbation_tokenizer.tokenizer),
        )

        with torch.no_grad():
            first_batch = next(iter(dataloader))
            input_ids = first_batch["input_ids"].to(self.device)
            attention_mask = first_batch["attention_mask"].to(self.device)

            num_labels = self.model(
                input_ids, attention_mask=attention_mask
            ).logits.size(1)
            if aggregation == "hard":
                outputs = torch.zeros(num_labels, dtype=torch.int32, device=self.device)
            else:
                outputs = torch.zeros(num_labels, device=self.device)

        for batch in dataloader:
            input_ids, attention_mask = batch["input_ids"].to(self.device), batch[
                "attention_mask"
            ].to(self.device)
            logits = self.model(input_ids, attention_mask=attention_mask).logits

            if aggregation == "hard":
                preds = logits.argmax(dim=1)
                outputs.index_add_(0, preds, torch.ones_like(preds, dtype=torch.int32))
            elif aggregation == "soft":
                probs = logits.softmax(dim=1)
                outputs += probs.sum(dim=0)
            else:
                outputs += logits.sum(dim=0)

        self.train(training)
        if normalize:
            return outputs.double() / num_samples
        return outputs

    def _certified_radius(
        self,
        text: str,
        num_samples: int = 1000,
        alpha: float = 0.05,
        aggregation: str = "hard",
        batch_size: int = 32,
        **kwargs,
    ):
        if aggregation != "hard":
            raise NotImplementedError("Only 'hard' aggregation is supported.")
        with torch.no_grad():
            label_counts = (
                self.agg_repeat_forward(
                    text, num_samples, aggregation=aggregation, batch_size=batch_size
                )
                .cpu()
                .numpy()
            )
        cr_pred, certified_radius = self.perturbation_tokenizer.certified_radius(
            text, counts=label_counts, alpha=alpha, **kwargs
        )
        return CertifiedPrediction(
            certified_radius=certified_radius,
            cr_label_counts=label_counts,
            cr_pred=cr_pred,
        )

    def predict(
        self,
        text: str,
        num_samples: int = 100,
        aggregation: str = "hard",
        batch_size: int = 32,
        **kwargs,
    ):
        if aggregation != "hard":
            raise NotImplementedError("Only 'hard' aggregation is supported.")
        with torch.no_grad():
            label_counts = (
                self.agg_repeat_forward(
                    text, num_samples, aggregation=aggregation, batch_size=batch_size
                )
                .cpu()
                .numpy()
            )
        pred, pval = self.perturbation_tokenizer.predict(
            text, counts=label_counts, **kwargs
        )
        return CertifiedPrediction(
            pred=pred, pred_label_counts=label_counts, pred_pval=pval
        )

    def certify(
        self,
        text: str,
        pred_num_samples: int = 100,
        pred_kwargs: dict = {},
        cr_num_samples: int = 1000,
        cr_kwargs: dict = {},
        batch_size: int = 32,
    ):
        pred = self.predict(
            text=text,
            num_samples=pred_num_samples,
            batch_size=batch_size,
            **pred_kwargs,
        )
        if cr_num_samples > 0:
            pred += self._certified_radius(
                text=text,
                num_samples=cr_num_samples,
                batch_size=batch_size,
                **cr_kwargs,
            )
        return pred

    def train(self, *args, **kwargs):
        self.model.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.model.eval(*args, **kwargs)

    @property
    def training(self):
        return self.model.training


def certified_predictions_to_dataframe(
    certified_predictions: List[CertifiedPrediction],
) -> pd.DataFrame:
    # Initialize lists to store data for each field in the CertifiedPrediction dataclass
    all_preds = []
    all_pred_label_counts = []
    all_pred_pvals = []
    all_certified_radii = []
    all_cr_label_counts = []
    all_cr_preds = []

    # Populate the lists from the CertifiedPrediction objects
    for cp in certified_predictions:
        cp_dict = asdict(cp)
        all_preds.append(cp_dict.get("pred"))
        all_pred_label_counts.append(cp_dict.get("pred_label_counts"))
        all_pred_pvals.append(cp_dict.get("pred_pval"))
        all_certified_radii.append(cp_dict.get("certified_radius"))
        all_cr_label_counts.append(cp_dict.get("cr_label_counts"))
        all_cr_preds.append(cp_dict.get("cr_pred"))

    # Determine the number of classes from either all_pred_label_counts or all_cr_label_counts
    num_labels = None
    if any(x is not None for x in all_pred_label_counts):
        num_labels = len(next(x for x in all_pred_label_counts if x is not None))
    elif any(x is not None for x in all_cr_label_counts):
        num_labels = len(next(x for x in all_cr_label_counts if x is not None))

    # Initialize DataFrame with placeholder values, if num_labels could be determined
    if num_labels is not None:
        label_count_column_names = [f"label_{i}_count" for i in range(num_labels)]
        cr_label_count_column_names = [f"cr_label_{i}_count" for i in range(num_labels)]
        df = pd.DataFrame(
            columns=label_count_column_names
            + cr_label_count_column_names
            + [
                "pred",
                "pred_pval",
                "certified_radius",
                "cr_pred",
            ]
        )
    else:
        df = pd.DataFrame()

    # Populate the DataFrame with actual values, conditionally
    df["pred"] = all_preds
    df["pred_pval"] = all_pred_pvals
    df["certified_radius"] = all_certified_radii
    df["cr_pred"] = all_cr_preds

    if num_labels is not None:
        df[label_count_column_names] = pd.DataFrame(
            all_pred_label_counts
            if any(x is not None for x in all_pred_label_counts)
            else None,
            columns=label_count_column_names,
        )
        df[cr_label_count_column_names] = pd.DataFrame(
            all_cr_label_counts
            if any(x is not None for x in all_cr_label_counts)
            else None,
            columns=cr_label_count_column_names,
        )
    return df
