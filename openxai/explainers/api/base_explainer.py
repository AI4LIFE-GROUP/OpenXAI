import torch
from abc import ABC, abstractmethod


class Explainer(ABC):
    """
    Abstract class to implement custom explanation methods for a given.
    Parameters
    ----------
    mlmodel: xai-bench.models.MLModel
        Classifier we wish to explain.
    Methods
    -------
    get_explanations:
        Generate explanations for given input.
    Returns
    -------
    None
    """

    def __init__(self, mlmodel):
        self.model = mlmodel

    @abstractmethod
    def get_explanation(self, inputs: torch.tensor, label: torch.Tensor):
        """
        Generate explanations for given input/s.
        Parameters
        ----------
        inputs: torch.tensor
            Input in two-dimensional shape (m, n).
        label: torch.tensor
            Label
        Returns
        -------
        torch.tensor
            Explanation vector/matrix.
        """
        pass
