import torch
from ...api import Explainer
from captum.attr import InputXGradient as InputXGradient_Captum


class InputTimesGradient(Explainer):
    """
    A baseline approach for computing the attribution.
    It multiplies input with the gradient with respect to input.
    https://arxiv.org/abs/1605.01713
    """

    def __init__(self, model):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """
        super(InputTimesGradient, self).__init__(model)

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

        input_x_gradient = InputXGradient_Captum(self.model)

        attribution = input_x_gradient.attribute(x, target=label)

        return attribution
