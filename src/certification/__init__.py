"""
CERTified Edit Distance defense (CERT-ED) authors authored this file

ChatGPT and/or Copilot are used in generating scaffolding code for this file
"""
from .masking_mech import MaskingMech
from .edit_mech import EditMech

from .perturbation import BasePerturbation, NullPerturbation, PerturbationTokenizer
from .smoothed_classifier import SmoothedClassifierWrapper, certified_predictions_to_dataframe
from .var_del_len_mech import VarDelLenMech
from .var_del_mech import VarDelMech
from .var_del_optim import VarDelOptim, BinnedLenAgent

def deletion_perturbation(*args, **kwargs):
    kwargs["p_ins"] = kwargs["p_sub"] = 0
    return EditMech(*args, **kwargs)

def inssub_perturbation(*args, **kwargs):
    kwargs["p_del"] = 0
    return EditMech(*args, **kwargs)

def sub_perturbation(*args, **kwargs):
    kwargs["p_ins"] = kwargs["p_del"] = 0
    return EditMech(*args, **kwargs)

perturbation_tokenizers = {
    "Base": NullPerturbation,
    "MaskingMech": MaskingMech,
    "DeletionMech": deletion_perturbation,
    "SubInsMech": inssub_perturbation,
    "SubMech": sub_perturbation,
    #"EditMech": EditMech,
    "VarDelLenMech": VarDelLenMech,
    "VarDelOptim": VarDelOptim,
}
