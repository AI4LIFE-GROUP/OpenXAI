import torch
from ...api import Explainer
from captum.attr import IntegratedGradients as IG_Captum


class IntegratedGradients(Explainer):
    """
    Provides integrated gradient attributions.
    Original paper: https://arxiv.org/abs/1703.01365
    """

    def __init__(self, model, method: str = 'gausslegendre', multiply_by_inputs: bool = False, baseline=None) -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """

        self.method = method
        self.multiply_by_inputs = multiply_by_inputs
        self.baseline = baseline

        super(IntegratedGradients, self).__init__(model)

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

        ig = IG_Captum(self.model, self.multiply_by_inputs)

        attribution = ig.attribute(x, target=label, method=self.method, baselines=self.baseline)

        return attribution
