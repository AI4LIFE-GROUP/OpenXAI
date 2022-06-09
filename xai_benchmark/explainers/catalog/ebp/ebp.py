import torch
from ...api import Explainer
from torchray.attribution.excitation_backprop import excitation_backprop


class EBP(Explainer):
    """
    Provides excitation backpropagation.
    Excitation backprop paper: https://arxiv.org/abs/1608.00507
    TorchRay reference: https://facebookresearch.github.io/TorchRay/_modules/torchray/attribution/excitation_backprop.html
    """
    
    def __init__(self, model):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """
        super(EBP, self).__init__(model)

    def get_explanation(self, x: torch.Tensor, label: torch.Tensor):
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
        
        attribution = excitation_backprop(
            self.model,
            x,
            label
        )

        return attribution
