import torch
from ...api import Explainer
from captum.attr import LRP as LRP_Captum


class LRP(Explainer):
    """
    Provides layer-wise relevance propagation explanations with respect to the input layer.
    https://arxiv.org/abs/1604.00825
    """
    
    def __init__(self, model) -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """
        super(LRP, self).__init__(model)

    def get_explanation(self, x: torch.Tensor, label: torch.Tensor) -> torch.tensor:
        """
        Explain an instance prediction.
        Args:
            x (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance): feature tensor
            label (torch.Tensor, [N x ...]): labels to explain
        Returns:
            exp (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance: instance level explanation):
        """
        self.model.eval()
        self.model.zero_grad()
        
        lrp = LRP_Captum(self.model)
        
        attribution = lrp.attribute(x, target=label)

        return attribution
