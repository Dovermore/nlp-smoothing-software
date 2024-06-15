"""
CERTified Edit Distance defense (CERT-ED) authors authored this file

ChatGPT and/or Copilot are used in generating scaffolding code for this file
"""
from typing import Optional, Sequence

import numpy as np

from .perturbation import PerturbationTokenizer, Tokens
from .utils import StagedEdits, topk_ci
from .edit_certs import SubInsMechCert, DeletionMechCert


def sample_subins(
    x: np.ndarray,
    p_sub: float,
    p_ins: float,
    ins_locs: Optional[np.ndarray] = None,
    sub_locs: Optional[np.ndarray] = None,
    min_keep: int = 2,
) -> StagedEdits:
    """Sample sub/ins masking edits

    Args:
        x: sequence to edit
        p_sub: probability of applying a substitution at a permitted location
        p_ins: controls the expected number of insertions (given by `p_ins / (1 - p_ins)`) at a permitted location
        ins_locs: zero-indexed locations in `x` before which insertions are permitted
        sub_locs: zero-indexed locations in `x` where substitutions are permitted.

    Returns:
        Random edits to apply
    """
    if sub_locs is None:
        sub_locs = np.arange(x.size, dtype=int)
    if ins_locs is None:
        ins_locs = np.arange(x.size + 1, dtype=int)

    # Decide where to apply substitutions by flipping a biased coin at each permitted location
    rnd_sub_locs = sub_locs[np.random.uniform(size=sub_locs.size) < p_sub]
    if rnd_sub_locs.size == sub_locs.size:
        min_keep = min(min_keep, rnd_sub_locs.size)
        # Randomly keep two locations to avoid an completely maske sequence (consistent with RanMASK)
        rnd_sub_locs = np.delete(rnd_sub_locs, np.random.choice(rnd_sub_locs.size, min_keep, replace=False))

    if p_ins > 0:
        # Insertions not permitted before a substitution
        ins_locs = np.setdiff1d(ins_locs, rnd_sub_locs, assume_unique=True)
        # Decide how many insertions to apply before each permitted location
        counts_ins_locs = np.random.geometric(1 - p_ins, size=ins_locs.size) - 1
        rnd_ins_locs = np.repeat(ins_locs, counts_ins_locs)
    else:
        rnd_ins_locs = np.empty(0, dtype=int)

    return StagedEdits(sub_locs=rnd_sub_locs, ins_locs=rnd_ins_locs)


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


class EditMech(PerturbationTokenizer):
    POST_PROCESSING_OPTIONS = {"remove", "unique", "none"}
    SMOOTHING_OPTIONS = {"deletion", "sub", "inssub"}

    def __init__(
        self,
        p_del: float = 0,
        p_sub: float = 0,
        p_ins: float = 0,
        post_processing: str = "none",
        **kwargs,
    ) -> None:
        """
        Args:
            p_sub: probability of applying a substitution at a location.
            p_ins: probability controlling the expected number of insertions (given by `p_ins / (1 - p_ins)`) at a
                location.
            post_processing: post-processing method to apply. Must be one of:
                    - "none": no post-processing is applied,
                    - "unique": consecutive masked values are collapsed to a single masked value,
                    - "remove": masked values are removed.
                Defaults to "remove".
        """
        super().__init__(**kwargs)
        self.post_processing = self.__check_post_processing(post_processing)
        self.smoothing = self.__check_smoothing(p_del, p_sub, p_ins)
        self.mask_value = self.tokenizer.mask_token
        self.p_del = p_del
        self.p_sub = p_sub
        self.p_ins = p_ins

    @classmethod
    def __check_post_processing(cls, post_processing):
        post_processing = post_processing.lower()
        if post_processing not in cls.POST_PROCESSING_OPTIONS:
            raise ValueError(
                "Unknown option %s for postprocessing, options are: %s"
                % (post_processing, cls.POST_PROCESSING_OPTIONS)
            )
        return post_processing

    @classmethod
    def __check_smoothing(cls, p_del, p_sub, p_ins):
        if p_del > 0 and p_sub == 0 and p_ins == 0:
            return "deletion"
        elif p_del == 0 and p_sub > 0 and p_ins == 0:
            return "sub"
        elif p_del == 0 and p_sub > 0 and p_ins > 0:
            return "subins"
        else:
            raise ValueError(
                "Unknown combinations of probabilities, you need to specify combinations thats either {deletion, sub or subins}"
            )

    def perturb_tokens(self, tokens: Tokens) -> Tokens:
        tokens = np.array(tokens, dtype="<U32")
        # No need to randomly insert masked values if they"re going to be removed afterwards
        p_del = self.p_del
        p_sub = self.p_sub
        p_ins = 0.0 if self.post_processing == "remove" else self.p_ins
        if self.smoothing == "deletion":
            edits = sample_del(tokens, p_del=p_del, del_locs=None)
        else:
            edits = sample_subins(
                tokens,
                p_sub=p_sub,
                p_ins=p_ins,
                ins_locs=None,
                sub_locs=None,
                min_keep=2,
            )

        if self.post_processing == "remove":
            # Equivalent to deleting masked values *after* applying the random edits (but more efficient)
            edits.del_locs = edits.sub_locs
            edits.sub_locs = []

        perturbed_tokens = edits.apply(
            tokens, del_value=None, ins_value=self.mask_value, sub_value=self.mask_value
        )

        if self.post_processing == "unique":
            # Replace consecutive runs of masked values by a single masked value
            masked = np.where(perturbed_tokens == self.mask_value)[0]
            df = np.diff(masked, prepend=-2)
            perturbed_tokens = np.delete(perturbed_tokens, masked[df == 1])
        return perturbed_tokens.tolist()

    def __certified_radius_deletion(
        self,
        input,
        top1_lb,
        top2_ub,
        threat_model: str = "edit",
        **kwargs,
    ) -> float:
        p_del = self.p_del
        if threat_model in ("edit", "sub", "inssub", "delsub"):
            radius = DeletionMechCert.edit_cert(p_del=p_del, top1=top1_lb, top2=top2_ub)
        elif threat_model in ("ins",):
            radius = DeletionMechCert.ins_cert(p_del=p_del, top1=top1_lb, top2=top2_ub)
        elif threat_model in ("del", "delins"):
            radius = DeletionMechCert.del_cert(p_del=p_del, top1=top1_lb, top2=top2_ub)
        else:
            raise ValueError("Unknown threat model: {}".format(threat_model))
        return radius

    def __certified_radius_subins(
        self,
        input,
        top1_lb,
        top2_ub,
        threat_model: str = "edit",
        **kwargs,
    ) -> float:
        """Compute the certified edit distance radius for inputs to a classifier smoothed under this perturbation

        Args:
            input: Unperturbed input sample.
            pred: Estimated prediction of the smoothed classifier for `input`. Must be a class index in the set
                {0, 1, 2, ..., n_classes - 1}.
            counts: Class frequencies for randomly perturbed inputs passed through the classifier. Must be a sequence
                where `counts[i]` is the number of perturbed inputs with class index `i`.

        Keyword args:
            alpha: Significance level. Defaults to 0.05.
            stat_test: Statistical test used to compute the certificate. Currently only "lee" is supported.

        Returns:
            The certified radius for this sample.
        """
        if threat_model in ("edit",):
            radius = SubInsMechCert.edit_cert(p_sub=self.p_sub, p_ins=self.p_ins, top1=top1_lb, top2=top2_ub)
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
        if stat_test.lower() == "cohen":
            [[top1, top1_lb, _]] = topk_ci(counts, alpha=alpha, k=1)
        elif stat_test.lower() == "lecuyer":
            [[top1, top1_lb, _], [_, _, top2_ub]] = topk_ci(counts, alpha=alpha, k=2)
        else:
            raise ValueError("Unknown statistical test: {}".format(stat_test))

        if self.smoothing == "deletion":
            radius = self.__certified_radius_deletion(
                input, top1_lb, top2_ub, alpha=alpha, **kwargs
            )
        elif self.smoothing == "sub":
            radius = self.__certified_radius_sub(
                input, top1_lb, top2_ub, alpha=alpha, **kwargs
            )
        elif self.smoothing == "subins":
            radius = self.__certified_radius_subins(
                input, top1_lb, top2_ub, alpha=alpha, **kwargs
            )
        else:
            raise ValueError("Unknown smoothing method: {}".format(self.smoothing))
        return top1, radius

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(smoothing={self.smoothing}, p_del={self.p_del}, p_sub={self.p_sub}, p_ins={self.p_ins}, mask_value={self.mask_value}, post_processing={self.post_processing})"
        )

    def __certified_radius_sub(
        self,
        input: str,
        pred: int,
        counts: Sequence[int],
        alpha: float = 0.05,
        stat_test: str = "cohen",
        strategy: str = "binary_search",
        **kwargs,
    ) -> float:
        raise NotImplementedError("Substitution certified radius not implemented yet")
