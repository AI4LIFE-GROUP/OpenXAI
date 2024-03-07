import torch
from ...api import BaseExplainer


class RandomBaseline(BaseExplainer):
    """
    A control baseline that returns a random explanation sampled independently of the 
    input and predictive model.
    """

    def __init__(self, model, seed=None) -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """
        self.seed = seed

        super(RandomBaseline, self).__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None) -> torch.tensor:
        """
        Returns a random standard normal vector of shape x.shape.
        
        Args:
            x (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance): feature tensor
            label (torch.Tensor, [N x ...]): labels to explain
        Returns:
            exp (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance: instance level explanation):
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
        attribution = torch.randn(size=x.shape)

        return attribution
