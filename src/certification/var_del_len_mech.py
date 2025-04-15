"""
AdaptDel authors authored this file with the help of ChatGPT
"""
# Aim for generalized class that includes del, inssub, sub
from typing import Optional, Sequence

import numpy as np

from .perturbation import PerturbationTokenizer, Tokens
from .utils import StagedEdits, topk_ci
from .var_del_len_certs import VarDelLenCert


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


class VarDelLenMech(PerturbationTokenizer):
    def __init__(
        self,
        k_keep: float = 20,
        p_del: float = 1,
        p_del_lb: float = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.k_keep = k_keep
        self.p_del = p_del
        self.p_del_lb = p_del_lb
        self.mask_value = self.tokenizer.mask_token
        assert 0 <= self.p_del <= 1, "Deletion probability must be in [0, 1]"
        assert self.k_keep >= 0, "Number of tokens to keep must be non-negative"

    def __compute_p_del(self, size) -> float:
        return max(self.p_del_lb, self.p_del - self.k_keep / size)

    def perturb_tokens(self, tokens: Tokens) -> Tokens:
        tokens = np.array(tokens, dtype="<U32")
        p_del_x = self.__compute_p_del(tokens.size)
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
        if threat_model in ("edit", "sub", "inssub", "delsub"):
            radius = VarDelLenCert.edit_cert(
                tokens,
                p_del=self.p_del,
                k_keep=self.k_keep,
                top1=top1_lb,
                top2=top2_ub,
                p_del_lb=self.p_del_lb,
            )
        elif threat_model in ("ins",):
            radius = VarDelLenCert.ins_cert(
                tokens,
                p_del=self.p_del,
                k_keep=self.k_keep,
                top1=top1_lb,
                top2=top2_ub,
                p_del_lb=self.p_del_lb,
            )
        elif threat_model in ("del", "delins"):
            radius = VarDelLenCert.del_cert(
                tokens,
                p_del=self.p_del,
                k_keep=self.k_keep,
                top1=top1_lb,
                top2=top2_ub,
                p_del_lb=self.p_del_lb,
            )
        else:
            raise ValueError("Unknown threat model: {}".format(threat_model))
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
            + f"(var_del_length, p_del={self.p_del}, k_keep={self.k_keep})"
        )
