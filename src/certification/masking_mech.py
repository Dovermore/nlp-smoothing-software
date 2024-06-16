"""
CERTified Edit Distance defense (CERT-ED) authors edited this file

Some codes are from the RS-Del code repository
"""
from math import ceil, comb, nan
from typing import Callable, Optional, Sequence

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from .perturbation import PerturbationTokenizer, Tokens
from .utils import binary_search_solve, brute_force_solve, combln, topk_ci
from .masking import mask_sentence


def _lecuyer_cert(
    num_dim: int,
    num_mask: int,
    counts: Sequence[int],
    alpha: float = 0.05,
) -> Callable[[int], float]:
    """Approximate certificate for randomized masking based on the statistical test of Lecuyer et al. (2019)

    Args:
        num_dim: Number of dimensions in the input.
        num_mask: Number of dimensions randomly masked in the input.
        counts: Class frequencies for randomly perturbed inputs passed through the classifier. Must be a sequence
            where `counts[i]` is the number of perturbed inputs with class index `i`.

    Keyword args:
        alpha: Significance level. Defaults to 0.05.

    Returns:
        A decreasing function of the L0 radius. If function is positive for a particular radius, then the certificate
        holds.
    """
    [[_, p_A_lower, _], [_, _, p_B_upper]] = topk_ci(counts, alpha=alpha, k=2)

    def f(radius: int) -> float:
        delta = 1.0 - np.exp(
            combln(num_dim - radius, num_mask - radius) - combln(num_dim, num_mask)
        )
        return 0.5 * (p_A_lower - p_B_upper) - delta

    return f


def _cohen_cert(
    num_dim: int,
    num_mask: int,
    counts: Sequence[int],
    threshold: Optional[float] = None,
    alpha: float = 0.05,
) -> Callable[[int], float]:
    """Approximate certificate for randomized masking based on the statistical test presented in Section 3.2.2 of
    Cohen, Rosenfeld and Kolter (2019)

    Args:
        num_dim: Number of dimensions in the input.
        num_mask: Number of dimensions randomly masked in the input.
        counts: Class frequencies for randomly perturbed inputs passed through the classifier.

    Keyword args:
        alpha: Significance level. Defaults to 0.05.
        threshold: Classification threshold. Should be set to 0.5 for multiclass problems, but may be adjusted for
            two-class problems.

    Returns:
        A decreasing function of the L0 radius. If function is positive for a particular radius, then the certificate
        holds.
    """
    if threshold is None:
        threshold = 0.5

    [[_, p_A_lower, _]] = topk_ci(counts, alpha=alpha, k=1)

    def f(radius: int) -> float:
        delta = 1.0 - np.exp(
            combln(num_dim - radius, num_mask - radius) - combln(num_dim, num_mask)
        )
        return p_A_lower - threshold - delta

    return f


def _jia_cert(
    num_dim: int,
    num_mask: int,
    counts: Sequence[int],
    threshold: Optional[float] = None,
    alpha: float = 0.05,
) -> Callable[[int], float]:
    """Approximate certificate for randomized masking based on the method proposed by Jia et al. (2022)

    Args:
        num_dim: Number of dimensions in the input.
        num_mask: Number of dimensions randomly masked in the input.
        counts: Class frequencies for randomly perturbed inputs passed through the classifier.

    Keyword args:
        alpha: Significance level. Defaults to 0.05.
        threshold: Classification threshold. Should be set to 0.5 for multiclass problems, but may be adjusted for
            two-class problems.

    Returns:
        A decreasing function of the L0 radius. If function is positive for a particular radius, then the certificate
        holds.
    """
    # We implement Jia et al.'s certificate for the 2-class setting, where their method for estimating bounds on the
    # probability scores is the same as Cohen et al.'s. We therefore raise an exception if we're not in the 2-class
    # setting. Note: if we want to extend to multi-class, we need to implement SimuEM (Jia et al., 2020).
    if len(counts) != 2:
        raise NotImplementedError(
            "This certificate is not yet implemented for more than 2 classes"
        )

    # One-sided lower bound on probability of most frequent class (\underbar{p_A}).
    max_counts = np.max(counts)
    num_samples = np.sum(counts)
    p_A_lower, _ = proportion_confint(
        max_counts, num_samples, alpha=2 * alpha, method="beta"
    )

    # Adjust the lower bound by rounding up to the nearest integer multiple of q = 1 / comb(num_dim, num_mask).
    # We only need to do the rounding if q >= machine epsilon, otherwise it has no impact due to floating point
    # quantization.
    machine_eps = np.finfo(float).eps
    if combln(num_dim, num_mask) <= -np.log(machine_eps):
        # Use arbitrary precision integer arithmetic
        q_inv = comb(num_dim, num_dim - num_mask)
        # Integer representation of lower bound
        num, den = p_A_lower.as_integer_ratio()
        # Below is equivalent to p_A_lower = np.ceil(p_A_lower * q_inv) / q_inv. By adding `den - 1`, we perform
        # integer division where the result is rounded up, rather than down.
        p_A_lower = ((q_inv * num + den - 1) // den) / q_inv

    def f(radius: int) -> float:
        delta = 1.0 - np.exp(
            combln(num_dim - radius, num_mask - radius) - combln(num_dim, num_mask)
        )
        return p_A_lower - threshold - delta

    return f


class MaskingMech(PerturbationTokenizer):
    """Masking randomized smoothing mechanism"""

    def __init__(
        self,
        mask_fraction: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mask_fraction = mask_fraction
        self.mask_value = self.tokenizer.mask_token

    def perturb_tokens(self, input: Tokens) -> Tokens:
        sentence = " ".join(input)
        masked_sentence = mask_sentence(
            sentence,
            self.mask_fraction,
            self.mask_value,
            tokenization=self.tokenization,
            min_keep=0,
        )[0]
        return masked_sentence.split(" ")

    def certified_radius(
        self,
        input: str,
        counts: np.array,
        alpha: float = 0.05,
        stat_test: str = "cohen",
        strategy: str = "binary_search",
        **kwargs,
    ) -> float:
        """Compute the certified radius for inputs to a classifier smoothed under this perturbation

        Args:
            input: Unperturbed binary sample. It must contain metadata with an entry for 'insn_addr'.
            pred: Estimated prediction of the smoothed classifier for `input`. Must be a class index in the set
                {0, 1, 2, ..., n_classes - 1}.
            counts: Class frequencies for randomly perturbed inputs passed through the classifier. Must be a sequence
                where `counts[i]` is the number of perturbed inputs with class index `i`.

        Keyword args:
            alpha: Significance level. Defaults to 0.05.
            stat_test: Statistical test used to compute the certificate. If "lecuyer" the test is based on
                Proposition 2 of Lecuyer et al. (2019). If "cohen" the test is based on Section 3.2.2 of Cohen,
                Rosenfeld and Kolter (2019).

        Returns:
            Return certified radius for this sample.
        """
        valid_stat_tests = {"cohen", "lecuyer", "jia"}
        if not stat_test in valid_stat_tests:
            raise ValueError(
                "`stat_test = {}` is not one of the permitted values {}".format(
                    stat_test, valid_stat_tests
                )
            )
        num_chunks = len(input.split())
        num_mask = ceil(self.mask_fraction * num_chunks)

        # Handle file with no instructions
        if num_mask == 0:
            return nan
        threshold = self.threshold if self.threshold is not None else 0.5
        if stat_test == "lecuyer":
            if self.threshold != 0.5:
                raise ValueError("lecuyer stat_test cannot be used if threshold != 0.5")
            f = _lecuyer_cert(num_chunks, num_mask, counts, alpha=alpha)
        elif stat_test == "cohen":
            f = _cohen_cert(
                num_chunks,
                num_mask,
                counts,
                threshold=threshold,
                alpha=alpha,
            )
        else:
            f = _jia_cert(
                num_chunks,
                num_mask,
                counts,
                threshold=threshold,
                alpha=alpha,
            )

        if strategy == "brute_force":
            largest_radius = brute_force_solve(f, x_max=num_mask)
        elif strategy == "binary_search":
            largest_radius = binary_search_solve(f, x_max=num_mask)
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")

        return np.argmax(counts), float(largest_radius)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(mask_fraction={self.mask_fraction}, mask_value={self.mask_value}, threshold={self.threshold})"
        )
