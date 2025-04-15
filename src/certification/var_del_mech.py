"""
AdaptDel authors authored this file with the help of ChatGPT
"""
# Aim for generalized class that includes del, inssub, sub
from typing import Optional, Sequence, Callable

import numpy as np

from .perturbation import PerturbationTokenizer, Tokens
from .utils import StagedEdits, topk_ci
from .var_del_certs import find_max_radius


def sample_del(
    x: np.ndarray,
    p_del: float,
    del_locs: Optional[np.ndarray] = None,
) -> StagedEdits:
    if del_locs is None:
        del_locs = np.arange(x.size, dtype=int)
    # Decide where to apply substitutions by flipping a biased coin at each permitted location
    rnd_del_locs = del_locs[np.random.uniform(size=del_locs.size) < p_del]
    return StagedEdits(del_locs=rnd_del_locs)


class VarDelMech(PerturbationTokenizer):
    def __init__(
        self,
        f_del: Callable,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.f_del = f_del
        self.mask_value = self.tokenizer.mask_token

    def __compute_p_del(self, tokens) -> float:
        size = tokens.size
        return self.f_del(size)

    def perturb_tokens(self, tokens: Tokens) -> Tokens:
        tokens = np.array(tokens, dtype="<U32")
        p_del_x = self.__compute_p_del(tokens)
        edits = sample_del(tokens, p_del=p_del_x, del_locs=None)
        perturbed_tokens = edits.apply(
            tokens, del_value=None, ins_value=self.mask_value, sub_value=self.mask_value
        )
        return perturbed_tokens.tolist()

    def __certified_radius_deletion(
        self,
        tokens,
        top1_lb,
        top2_ub,
        threat_model: str = "edit",
        **kwargs,
    ) -> float:
        if threat_model == "edit":
            plausible_subset = {"sub", "ins", "del"}
        else:
            plausible_subset = set()
            if "sub" in threat_model:
                plausible_subset.add("sub")
            if "ins" in threat_model:
                plausible_subset.add("ins")
            if "del" in threat_model:
                plausible_subset.add("del")
            if len(plausible_subset) == 0:
                raise ValueError("Unknown threat model: {}".format(threat_model))
        radius = find_max_radius(tokens, self.f_del, top1_lb, top2_ub, plausible_subset,)
        return radius

    def certified_radius(
        self,
        input: str,
        counts: np.array,
        alpha: float = 0.05,
        stat_test: str = "cohen",
        **kwargs,
    ) -> float:
        top1_lb, top2_ub = None, None
        if stat_test is None:
            if len(counts) == 2:
                stat_test = "cohen"
            else:
                stat_test = "lecuyer"
        if stat_test.lower() == "cohen":
            [[top1, top1_lb, _]] = topk_ci(counts, alpha=alpha, k=1)
        elif stat_test.lower() == "lecuyer":
            [[top1, top1_lb, _], [_, _, top2_ub]] = topk_ci(counts, alpha=alpha, k=2)
        else:
            raise ValueError("Unknown statistical test: {}".format(stat_test))
        tokens = self.tokenize_input(input)
        radius = self.__certified_radius_deletion(
            tokens, top1_lb, top2_ub, alpha=alpha, **kwargs
        )
        return top1, radius

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(var_del_length,)"
        )
