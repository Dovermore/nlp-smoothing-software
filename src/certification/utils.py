"""
CERTified Edit Distance defense (CERT-ED) authors authored this file

ChatGPT and/or Copilot are used in generating scaffolding code for this file
"""
from dataclasses import dataclass
from math import inf
from typing import Callable, Optional

import numpy as np
import torch
from scipy.special import gammaln
from torch.utils.data import Dataset
from statsmodels.stats.proportion import proportion_confint


def combln(n, k) -> float:
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def brute_force_solve(f: Callable[[int], float], x_max: Optional[int] = None) -> int:
    """Find the largest non-negative integer argument of a decreasing function, such that its output remains positive

    Note:
    The solution is found by brute force: testing each value of the argument starting from 0.

    Args:
        f: A real-valued decreasing function, whose domain is the non-negative integers.
        x_max: Upper bound on the domain of `f`.

    Returns:
        The argument that satisfies the constraint. If the function is never positive, a value of -1 is returned.
    """
    x = -1
    while (x + 1 <= x_max if x_max else True) and f(x + 1) > 0:
        x += 1
    return x


def _exponential_bound(
    f: Callable[[int], float], base: Optional[int] = 4, x_max: Optional[int] = None
) -> int:
    # This assumes f(-1) > 0
    if base <= 1:
        raise ValueError("Base value have to be larger than 1")
    x = 1
    while (x_max is None or x <= x_max) and f(x) > 0:
        x *= base
    if x_max:
        x = min(x, x_max)
    return x


def binary_search_solve(f: Callable[[int], float], x_max: Optional[int] = None) -> int:
    """Find the largest non-negative integer argument of a decreasing function, such that its output remains positive
    using binary search

    Note:
    The solution is found by binary search: the upper value is specified by x_max.

    Args:
        f: A real-valued decreasing function, whose domain is the non-negative integers.
        x_max: Upper bound on the domain of `f`.

    Returns:
        The argument that satisfies the constraint. If the function is never positive, a value of -1 is returned.
    """
    x_left, f_left = -1, 1
    x_max_bound = _exponential_bound(f, base=4, x_max=x_max)
    x_max = min(x_max_bound, inf if x_max is None else x_max)
    x_right, f_right = x_max, f(x_max)

    # The maximum value is still negative, return the left x value
    if f(x_left + 1) < 0:
        return x_left
    # The minimum value is still positive, return the right x value
    elif f_right > 0:
        return x_right

    # Stop when left = right - 1, return left
    while x_left < x_right - 1:
        x_mid = (x_right + x_left) // 2
        f_mid = f(x_mid)
        if f_mid <= 0:
            x_right, f_right = x_mid, f_mid
        elif f_mid > 0:
            x_left, f_left = x_mid, f_mid
        else:
            raise ValueError("Nan detected when computing")
    assert f_left > 0 and f_right <= 0 and x_left == x_right - 1, "BS error"
    return x_left

def string_to_tensor(input):
    ords = list(map(ord, input))
    return torch.tensor(ords, dtype=torch.int32)
    
def tensor_to_string(input):
    # Convert tensor to python list.
    ords = input.tolist()
    # Convert ordinal values to characters and join them into a string.
    return "".join(map(chr, ords))

@dataclass
class StagedEdits:
    """This class stores locations in a sequence where edits (deletions, substitutions and insertions) are to be made. 
    The locations are defined with respect to the original sequence, before any edits have been made.
    """
    del_locs: Optional[np.ndarray] = None
    sub_locs: Optional[np.ndarray] = None
    # An insertion at location `j`, means an element is inserted *before* the existing element at index `j` 
    # (similar to how `i` in vim works)
    ins_locs: Optional[np.ndarray] = None

    def apply(
        self, 
        x: np.ndarray, 
        sub_value: np.dtype, 
        ins_value: np.dtype,
        del_value: Optional[np.dtype] = None, 
    ) -> np.ndarray:
        """Apply staged edits

        Args:
            x: Sequence to edit. The elements are assumed to be non-negative integers.
            del_value: If an integer value is given, deletions are performed by substituting with this value. If no 
                value is specified, deletions are performed by removing elements from the sequence.
            ins_value: An integer value to insert when performing an insertion edit.
            sub_value: An integer value to substitute when performing a substitution edit.
        
        Returns:
            Edited sequence.
        """
        x_edit = x.copy()

        # Apply substitutions
        if self.sub_locs is not None:
            x_edit[self.sub_locs] = sub_value

        # Apply deletions by substituting del_value
        if self.del_locs is not None:
            placeholder = '[NAN]' if x.dtype.kind == 'U' else -1
            x_edit[self.del_locs] = placeholder if del_value is None else del_value

        # Apply insertions
        if self.ins_locs is not None:
            x_edit = np.insert(x_edit, self.ins_locs, ins_value)

        if del_value is None and self.del_locs is not None:
            x_edit = x_edit[x_edit != placeholder]

        return x_edit


class RepeatSampleDataset(Dataset):
    def __init__(self, text: str, perturbation_tokenizer, num_samples: int, max_length=256, padding="longest"):
        self.text = text
        self.padding = padding
        self.max_length = max_length
        self.perturbation_tokenizer = perturbation_tokenizer
        self.num_samples = num_samples
        self.cache = {}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Check if the sample for this index is cached
        if index in self.cache:
            return self.cache[index]
        
        # If not cached, generate and cache the perturbed sample
        out = self.perturbation_tokenizer(
            self.text,
            truncation=True,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        self.cache[index] = out
        return out

def topk_ci(counts, alpha=0.05, k=2):
    counts, num_samples = np.array(counts), np.sum(counts)
    sorted_indices = np.argsort(counts)[::-1]
    alpha = alpha * 2 / k
    out = []
    for idx in sorted_indices[:k]:
        lb, ub = proportion_confint(counts[idx], num_samples, alpha=alpha, method="beta")
        out.append((idx, lb, ub))
    return out
