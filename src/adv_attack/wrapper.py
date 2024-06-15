"""
CERTified Edit Distance defense (CERT-ED) authors authored this file

ChatGPT and/or Copilot are used in generating scaffolding code for this file
"""
import torch
import numpy as np

from textattack.models.wrappers import ModelWrapper
from textattack.datasets import Dataset
from certification import SmoothedClassifierWrapper


class SmoothedClassifierAttackWrapper(ModelWrapper):
    """A model wrapper queries a model with a list of text inputs.

    Classification-based models return a list of lists, where each sublist
    represents the model's scores for a given input.

    Text-to-text models return a list of strings, where each string is the
    output – like a translation or summarization – for a given input.
    """

    def __init__(
        self,
        smoothed_classifier: SmoothedClassifierWrapper,
        num_samples: int = 100,
        aggregation: str = "hard",
        batch_size: int = 32,
    ):
        super().__init__()
        self.model = smoothed_classifier
        self.num_samples = num_samples
        self.aggregation = aggregation
        self.batch_size = batch_size

    def __call__(self, text_input_list, **kwargs):
        scores = []
        with torch.no_grad():
            for text_input in text_input_list:
                scores.append(
                    self.model.agg_repeat_forward(
                        text_input,
                        num_samples=self.num_samples,
                        aggregation=self.aggregation,
                        batch_size=self.batch_size,
                        normalize=True,
                    ).cpu().numpy()
                )
        scores = np.stack(scores)
        return scores

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens."""
        raise NotImplementedError()

    def _tokenize(self, inputs):
        """Helper method for `tokenize`"""
        raise NotImplementedError()
