"""
AdaptDel authors authored this file with the help of ChatGPT
"""
import warnings
from typing import Optional, Union
import numpy as np
import scipy.special as sp
from functools import lru_cache
from .utils import binary_search_solve, logminusexp, combln


@lru_cache(maxsize=None)
def rev_binomial_log_prob(k, n, p):
    # Calculate log of binomial probability to avoid numerical instability
    log_comb = combln(
        n, k
    )  # sp.gammaln(n + 1) - sp.gammaln(k + 1) - sp.gammaln(n - k + 1)
    log_prob = log_comb + k * np.log(p) + (n - k) * np.log(1 - p)
    return log_prob


def combln(n, k) -> float:
    return sp.gammaln(n + 1) - sp.gammaln(k + 1) - sp.gammaln(n - k + 1)


def compute_lowerbound(tokens, p_del, k_keep, mu, n_sub, n_ins, n_del, p_del_lb=0):
    with warnings.catch_warnings():
        # Convert all warnings to exceptions
        warnings.simplefilter("error")

        try:
            x_size = len(tokens)
            z_star_size = x_size - n_del - n_sub
            xb_size = x_size + n_ins - n_del
            p_del_x = max(p_del_lb, p_del - k_keep / x_size)
            p_del_xb = max(p_del_lb, p_del - k_keep / xb_size)

            if p_del_x <= 0:
                return 0
            if p_del_xb <= 0:
                return 0
            if mu - 1 + p_del_x ** (x_size - z_star_size) <= 0:
                return 0

            def gen_partitions():
                # Determine initial partition direction based on p_del_x and p_del_xb
                partition = 0 if p_del_x >= p_del_xb else z_star_size
                while 0 <= partition <= z_star_size:
                    yield partition
                    partition += 1 if p_del_x >= p_del_xb else -1

            # Calculate threshold in log space to avoid overflow
            log_threshold = np.log(mu - 1 + p_del_x ** (x_size - z_star_size))
            log_accumulated_prob_x = -np.inf  # Accumulate in log-space
            log_accumulated_prob_xb = (
                -np.inf
            )  # Start with negative infinity for log-space summation

            # Precompute log ratio for efficiency
            log_p_ratio = (
                np.log(p_del_x)
                + np.log(1 - p_del_xb)
                - (np.log(p_del_xb) + np.log(1 - p_del_x))
            )

            for i in gen_partitions():
                # Calculate binomial_i in log-space
                log_binomial_i = rev_binomial_log_prob(i, z_star_size, 1 - p_del_x)

                # Sum in log-space for accumulated_prob_x
                log_accumulated_prob_x = np.logaddexp(
                    log_accumulated_prob_x, log_binomial_i
                )

                if log_accumulated_prob_x >= log_threshold:  # Compare in log-space
                    break

                # Sum in log-space for accumulated_prob_xb
                log_accumulated_prob_xb = np.logaddexp(
                    log_accumulated_prob_xb, i * log_p_ratio + log_binomial_i
                )

            # Add the remaining probability in log-space
            floor_term = (
                logminusexp(
                    np.logaddexp(log_threshold, log_binomial_i), log_accumulated_prob_x
                )
                - log_binomial_i
                + combln(z_star_size, i)
            )
            remaining_prob_log_term = (
                i * log_p_ratio
                + (
                    np.log(np.floor(np.exp(floor_term)))
                    if np.floor(np.exp(floor_term)) > 0
                    else -np.inf
                )
                - combln(z_star_size, i)
                + log_binomial_i
            )
            log_accumulated_prob_xb = np.logaddexp(
                log_accumulated_prob_xb, remaining_prob_log_term
            )
            log_accumulated_prob_xb = xb_size * np.log(p_del_xb) - x_size * np.log(p_del_x) + log_accumulated_prob_xb
            return max(0, np.exp(log_accumulated_prob_xb))

        except Warning as w:
            # Reraise the warning as an error with a custom message
            raise RuntimeError(
                f"A warning was encountered during computation: {str(w)}"
            )


def compute_upperbound(tokens, p_del, k_keep, mu, n_sub, n_ins, n_del, p_del_lb=0):
    with warnings.catch_warnings():
        # Convert all warnings to exceptions
        warnings.simplefilter("error")

        try:
            x_size = len(tokens)
            z_star_size = x_size - n_del - n_sub
            xb_size = x_size + n_ins - n_del
            p_del_x = max(p_del_lb, p_del - k_keep / x_size)
            p_del_xb = max(p_del_lb, p_del - k_keep / xb_size)

            # Have to return 1 if fails
            if p_del_x <= 0:
                return 1
            if p_del_xb <= 0:
                return 1
            if mu - 1 + p_del_x ** (x_size - z_star_size) <= 0:
                return 1

            def gen_partitions():
                # Determine initial partition direction based on p_del_x and p_del_xb
                partition = 0 if p_del_x < p_del_xb else z_star_size
                while 0 <= partition <= z_star_size:
                    yield partition
                    partition += 1 if p_del_x < p_del_xb else -1

            # Calculate threshold in log space to avoid overflow
            log_threshold = np.log(mu)
            log_accumulated_prob_x = -np.inf  # Accumulate in log-space
            log_accumulated_prob_xb = (
                -np.inf
            )  # Start with negative infinity for log-space summation

            # Precompute log ratio for efficiency
            log_p_ratio = (
                np.log(p_del_x)
                + np.log(1 - p_del_xb)
                - (np.log(p_del_xb) + np.log(1 - p_del_x))
            )

            for i in gen_partitions():
                # Calculate binomial_i in log-space
                log_binomial_i = rev_binomial_log_prob(i, z_star_size, 1 - p_del_x)

                # Sum in log-space for accumulated_prob_x
                log_accumulated_prob_x = np.logaddexp(
                    log_accumulated_prob_x, log_binomial_i
                )

                if log_accumulated_prob_x >= log_threshold:  # Compare in log-space
                    break

                # Sum in log-space for accumulated_prob_xb
                log_accumulated_prob_xb = np.logaddexp(
                    log_accumulated_prob_xb, i * log_p_ratio + log_binomial_i
                )

            # Add the remaining probability in log-space
            floor_term = (
                logminusexp(
                    np.logaddexp(log_threshold, log_binomial_i), log_accumulated_prob_x
                )
                - log_binomial_i
                + combln(z_star_size, i)
            )
            remaining_prob_log_term = (
                i * log_p_ratio
                + (
                    np.log(np.floor(np.exp(floor_term)))
                    if np.floor(np.exp(floor_term)) > 0
                    else -np.inf
                )
                - combln(z_star_size, i)
                + log_binomial_i
            )
            log_accumulated_prob_xb = np.logaddexp(
                log_accumulated_prob_xb, remaining_prob_log_term
            )
            log_accumulated_prob_xb = xb_size * np.log(p_del_xb) - x_size * np.log(p_del_x) + log_accumulated_prob_xb
            return min(1, np.exp(log_accumulated_prob_xb))

        except Warning as w:
            # Reraise the warning as an error with a custom message
            raise RuntimeError(
                f"A warning was encountered during computation: {str(w)}"
            )


def search_worst_case_bound(
    r,
    tokens,
    p_del,
    k_keep,
    top1,
    top2=None,
    plausible_subset={"sub", "ins", "del"},
    return_all=False,
    p_del_lb=0,
):
    """
    Searches for the optimal combination of n_sub, n_ins, and n_del to minimize the lowerbound.

    Parameters:
        input: The input sequence (list or string)
        p_del: Probability of deletion
        k_keep: Number of elements to keep
        top1: Top prob
        top2: Runner up prob
        r: Total number of operations to distribute across n_sub, n_ins, and n_del
        plausible_subset: Set of operations allowed, e.g., {'n_sub', 'n_ins', 'n_del'}

    Returns:
        min_lowerbound_value: The lowest computed lowerbound value
        optimal_combination: A dictionary with the values of n_sub, n_ins, n_del that achieve the minimum lowerbound
    """
    # Define possible ranges for each operation based on the subset
    range_sub = range(r + 1) if "sub" in plausible_subset else [0]
    range_ins = range(r + 1) if "ins" in plausible_subset else [0]
    # range_del = range(r + 1) if 'del' in plausible_subset else [0]

    min_lowerbound_value = float("inf")
    optimal_combination = None

    # Iterate over n_sub and n_ins, compute n_del directly
    for n_sub in range_sub:
        for n_ins in range_ins:
            n_del = r - n_sub - n_ins
            if n_del < 0 or n_del >= r:  # Ensure n_del is within range
                continue
            if "del" not in plausible_subset and n_del != 0:
                continue  # Skip if n_del is not allowed by plausible_subset

            # Compute the lowerbound for this combination
            top1_lower_bound = compute_lowerbound(
                tokens, p_del, k_keep, top1, n_sub, n_ins, n_del, p_del_lb
            )
            top2_upper_bound = (
                compute_upperbound(
                    tokens, p_del, k_keep, top2, n_sub, n_ins, n_del, p_del_lb
                )
                if top2 is not None
                else 0.5
            )
            lowerbound_value = top1_lower_bound - min(top2_upper_bound, 0.5)

            # Update if a new minimum is found
            if lowerbound_value < min_lowerbound_value:
                min_lowerbound_value = lowerbound_value
                optimal_combination = {"n_sub": n_sub, "n_ins": n_ins, "n_del": n_del}
    if return_all:
        return min_lowerbound_value, optimal_combination
    else:
        return min_lowerbound_value


def find_max_radius(
    tokens,
    p_del,
    k_keep,
    top1,
    top2=None,
    plausible_subset={"sub", "ins", "del"},
    p_del_lb=0,
) -> int:
    """
    Finds the maximum value of r for which the minimum lowerbound is >= 0.5.

    Parameters:
        input: The input sequence (list or string)
        p_del: Probability of deletion
        k_keep: Number of elements to keep
        top1: Prob for top 1 class
        top2: Prob for top 2 class
        plausible_subset: Set of operations allowed, e.g., {'sub', 'ins', 'del'}
        x_max: Optional maximum bound for r (if known)

    Returns:
        The largest r that satisfies the lowerbound >= 0.5.
    """
    # Define the objective function directly as a lambda
    objective_function = lambda r: search_worst_case_bound(
        r,
        tokens,
        p_del,
        k_keep,
        top1,
        top2=top2,
        plausible_subset=plausible_subset,
        return_all=False,
        p_del_lb=p_del_lb,
    )
    # Use binary_search_solve to find the maximum r for which the lowerbound is >= 0.5
    max_r = binary_search_solve(objective_function, x_max=len(tokens))
    return max_r


class VarDelLenCert:
    @staticmethod
    def edit_cert(
        input: Union[str, list],
        p_del: float,
        k_keep: int,
        top1: float,
        top2: Optional[float] = None,
        plausible_subset={"sub", "ins", "del"},
        p_del_lb: float = 0,
    ) -> int:
        """Approximate edit distance certificate with substitutions, insertions, and deletions allowed.

        Args:
            input: The input sequence (list or string).
            p_del: Probability of deleting a value at a location.
            k_keep: Number of elements to keep.
            top1: Predicted probability (corresponds to `mu`).
            top2: Predicted runner-up probability (for compatibility, not used in current implementation).
            plausible_subset: Set of allowed operations (default allows 'sub', 'ins', and 'del').

        Returns:
            Radius (r) that satisfies the certificate constraint.
        """
        # Use find_max_radius to get the optimal r
        return find_max_radius(
            input, p_del, k_keep, top1, top2, plausible_subset, p_del_lb=p_del_lb
        )

    @staticmethod
    def del_cert(
        input: Union[str, list],
        p_del: float,
        k_keep: int,
        top1: float,
        top2: Optional[float] = None,
        p_del_lb: float = 0,
    ) -> int:
        """Approximate deletion distance certificate with only deletions allowed.

        Args:
            input: The input sequence (list or string).
            p_del: Probability of deleting a value at a location.
            k_keep: Number of elements to keep.
            top1: Predicted probability (corresponds to `mu`).
            top2: Predicted runner-up probability (for compatibility, not used in current implementation).

        Returns:
            Radius (r) that satisfies the deletion-only certificate constraint.
        """
        # Only deletions are allowed, set plausible_subset accordingly
        return find_max_radius(
            input,
            p_del,
            k_keep,
            top1,
            top2,
            plausible_subset={"del"},
            p_del_lb=p_del_lb,
        )

    @staticmethod
    def ins_cert(
        input: Union[str, list],
        p_del: float,
        k_keep: int,
        top1: float,
        top2: Optional[float] = None,
        p_del_lb: float = 0,
    ) -> int:
        """Approximate insertion distance certificate with only insertions allowed.

        Args:
            input: The input sequence (list or string).
            p_del: Probability of deleting a value at a location.
            k_keep: Number of elements to keep.
            top1: Predicted probability (corresponds to `mu`).
            top2: Predicted runner-up probability (for compatibility, not used in current implementation).

        Returns:
            Radius (r) that satisfies the insertion-only certificate constraint.
        """
        # Only insertions are allowed, set plausible_subset accordingly
        return find_max_radius(
            input,
            p_del,
            k_keep,
            top1,
            top2,
            plausible_subset={"ins"},
            p_del_lb=p_del_lb,
        )
