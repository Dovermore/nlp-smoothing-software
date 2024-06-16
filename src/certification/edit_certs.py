"""
CERTified Edit Distance defense (CERT-ED) authors edited this file

Some codes are from the RS-Del code repository
"""
import numpy as np


# For subins
class SubInsMechCert:
    @staticmethod
    def edit_cert(
        p_sub: float,
        p_ins: float,
        top1: float,
        top2: float = None,
    ) -> float:
        if top2 is not None:
            raise NotImplementedError("Not implemented for top2")
        else:
            # One-sided lower bound on probabilities of most frequent class (\underbar{p_A})
            # By setting `method ="beta"` we are computing an exact Clopper-Pearson interval.
            radius = np.log(3 / 2 - top1) / np.log(min(p_sub, p_ins))
            radius = np.floor(radius).item()
            return max(radius, -1)


class DeletionMechCert:
    @staticmethod
    def edit_cert(p_del: float, top1: float, top2: float) -> float:
        """Approximate edit distance certificate when all types of edits are permitted.

        Args:
            p_del: probability of deleting a value at a location.
            top1: predicted probability
            top2: predicted runner-up probability
            eta: threshold for the prediction

        Returns:
            Radius of the certificate
        """
        if top2 is None:
            top2 = 1 - top1
            #log(1 + my'/2 - my/2) / log(p_del)
        radius = np.log((2 + top2 - top1) / 2) / np.log(p_del)
        radius = np.floor(radius).item()
        return max(radius, -1)

    @staticmethod
    def del_cert(p_del: float, top1: float, top2: float) -> float:
        """Approximate deletion distance certificate when all types of edits are permitted.

        Args:
            p_del: probability of deleting a value at a location.
            top1: predicted probability
            eta: threshold for the prediction

        Returns:
            Radius of the certificate
        """
        eta: float = 0.5
        if top2 is not None:
            raise NotImplementedError("Not implemented for top2")
        radius = np.log(eta / top1) / np.log(p_del)
        radius = np.floor(radius).item()
        return max(radius, -1)

    @staticmethod
    def ins_cert(p_del: float, top1: float, top2: float) -> float:
        """Approximate deletion insertion distance certificate when all types of edits are permitted.

        Args:
            p_del: probability of deleting a value at a location.
            top1: predicted probability
            eta: threshold for the prediction

        Returns:
            Radius of the certificate
        """
        eta: float = 0.5
        if top2 is not None:
            raise NotImplementedError("Not implemented for top2")
        radius = np.log((1 - top1) / (1 - eta)) / np.log(p_del)
        radius = np.floor(radius).item()
        return max(radius, -1)
