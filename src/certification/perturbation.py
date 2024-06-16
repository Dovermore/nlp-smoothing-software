"""
CERTified Edit Distance defense (CERT-ED) authors edited this file

Some codes are from the RS-Del code repository
"""
from typing import Optional, Sequence, Tuple, TypeVar, Generic, List, Any

from scipy.stats import binomtest
import numpy as np
from transformers import AutoTokenizer, BatchEncoding

IT = TypeVar("IT")
OT = TypeVar("OT")
Tokens = List[str]


class Transform(Generic[IT, OT]):
    def __call__(self, input: IT) -> OT:
        pass


def standard_predict(counts: np.ndarray, threshold: Optional[float] = None) -> Tuple[int, float]:
    num_labels = len(counts)
    if num_labels > 2 and threshold is not None:
        raise ValueError("Only supports explicit threshold for 2 class problems")

    if num_labels == 2:
        threshold = threshold if threshold is not None else 0.5
        counts_total = counts.sum()
        pred = int((counts[1] / counts_total > threshold))
        
        if pred:
            test = binomtest(counts[1], counts_total, p=threshold, alternative='greater')
        else:
            test = binomtest(counts[0], counts_total, p=1 - threshold, alternative='greater')
    
    else:
        toptwo_indices = np.argsort(-counts)[:2]  # Sort in descending order and take the first two indices
        toptwo_counts = counts[toptwo_indices]

        pred = toptwo_indices[0]
        
        n_A = toptwo_counts[0]
        n_B = toptwo_counts[1]
        test = binomtest(n_A, n_A + n_B, p=0.5, alternative='two-sided')

    return pred, test.pvalue


class BasePerturbation(Transform[IT, OT]):
    """Base class for a random perturbation"""

    def __init__(self, threshold: Optional[float] = None):
        super().__init__()
        self.threshold = threshold

    def __call__(self, input: IT) -> OT:
        """
        Args:
            input: Input to be perturbed.

        Returns:
            Perturbed output.
        """
        raise NotImplementedError("__call__ method is not defined for this class.")

    def __repr__(self):
        return self.__class__.__name__ + f"(threshold={self.threshold})"

    def predict(
        self,
        input: IT,
        counts: Sequence[int],
        **kwargs,
    ) -> Tuple[int, float]:
        """Compute the predicted class for an input to a classifier smoothed under this perturbation

        Args:
            input: Unperturbed input.
            counts: Class frequencies for randomly perturbed inputs passed through the classifier. Must be a sequence
                where `counts[i]` is the number of perturbed inputs with class index `i`.

        Keyword args:
            **kwargs: Other keyword arguments used in derived classes.

        Returns:
            The predicted class index and the p-value
        """
        return standard_predict(counts, self.threshold)

    def certified_radius(
        self, input: IT, counts: np.array, alpha: float = 0.05, **kwargs
    ) -> Tuple[int, float]:
        """Compute the certified radius for an input to a classifier smoothed under this perturbation

        Args:
            input: Unperturbed input.
            counts: Class frequencies for randomly perturbed inputs passed through the classifier. Must be a sequence
                where `counts[i]` is the number of perturbed inputs with class index `i`.

        Keyword args:
            alpha: Significance level. Defaults to 0.05.
            **kwargs: Other keyword arguments used in derived classes.

        Returns:
            Predicted class and the largest certified radius for the input.
        """
        raise NotImplementedError(
            "Certified radius method is not defined for this class."
        )


class PerturbationTokenizer(BasePerturbation[Any, BatchEncoding]):
    def __init__(
        self,
        tokenizer,
        threshold: Optional[float] = None,
        perturbation_on: bool = True,
        tokenization: str = "split",
        **kwargs,
    ):
        super().__init__(threshold=threshold, **kwargs)
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = tokenizer
        self.perturbation_on = perturbation_on
        self.tokenization = tokenization

    def perturb_tokens(self, tokens: Tokens) -> Tokens:
        raise NotImplementedError(
            "perturb_tokens method must be implemented in a subclass."
        )

    def __call__(self, input: str, *args, **kwargs) -> BatchEncoding:
        if self.tokenization == "tokenizer":
            tokens = self.tokenize(input)
        elif self.tokenization == "split":
            tokens = input.split()
        elif self.tokenization == "char":
            tokens = list(input)
        else:
            raise ValueError(f"Unknown tokenization method: {self.tokenization}")

        if self.perturbation_on:
            tokens = self.perturb_tokens(tokens)
        
        if self.tokenization == "split":
            tokens = self.tokenize(" ".join(tokens))
        elif self.tokenization == "char":
            tokens = self.tokenize("".join(tokens))

        output = self.convert_tokens_to_ids(tokens)
        output = self.prepare_for_model(output, *args, **kwargs)
        return output

    def tokenize(self, *args, **kwargs) -> Tokens:
        return self.tokenizer.tokenize(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def convert_ids_to_tokens(self, *args, **kwargs) -> Tokens:
        return self.tokenizer.convert_ids_to_tokens(*args, **kwargs)

    def prepare_for_model(self, *args, **kwargs) -> BatchEncoding:
        return self.tokenizer.prepare_for_model(*args, **kwargs)

    def decode(self, *args, **kwargs) -> str:
        return self.tokenizer.decode(*args, **kwargs)


class NullPerturbation(PerturbationTokenizer):
    """Null op perturbation"""

    def perturb_tokens(self, tokens: Tokens) -> Tokens:
        return tokens

    def certified_radius(
        self,
        input: str,
        counts: np.array,
        alpha: float = 0.05,
        **kwargs,
    ) -> float:
        pred = np.argmax(
            counts,
        )
        return pred, 0
