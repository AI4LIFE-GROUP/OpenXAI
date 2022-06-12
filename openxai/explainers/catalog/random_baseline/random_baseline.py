import torch
from ...api import Explainer


class RandomBaseline(Explainer):
    """
    A control baseline that returns a random explanation sampled independently of the 
    input and predictive model.
    """

    def __init__(self, model) -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """

        super(RandomBaseline, self).__init__(model)

    def get_explanation(self, x: torch.Tensor, label: torch.Tensor) -> torch.tensor:
        """
        Returns a random standard normal vector of shape x.shape.
        
        Args:
            x (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance): feature tensor
            label (torch.Tensor, [N x ...]): labels to explain
        Returns:
            exp (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance: instance level explanation):
        """
        attribution = torch.randn(size=x.shape)

        return attribution
