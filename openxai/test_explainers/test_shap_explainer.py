import xgboost
import shap
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from explainers.shap_explainer import SHAPExplainer
from shap import TreeExplainer
from shap import KernelExplainer
from shap import DeepExplainer


def test_shap_explainer():
    '''
    This test function is added to test the explanation model class for tabular and image data formats.
    '''

    # train an XGBoost model
    # X, y = shap.datasets.boston()
    # model = xgboost.XGBRegressor().fit(X, y)
    #
    # explainer  = SHAPExplainer(model, X, function_class = "tree")
    # print(explainer.get_explaination(X)[0])

    ## Checking the gradient explaination. Most of the code is taken from Shap package :https://github.com/slundberg/shap/blob/32701b120f09a74466e7072c3ef8b5a76550c43d/tests/explainers/test_gradient.py
    batch_size = 4
    class RandData:
        """ Ranomd data for testing.
        """

        def __init__(self, batch_size):
            self.current = 0
            self.batch_size = batch_size

        def __iter__(self):
            return self

        def __next__(self):
            self.current += 1
            if self.current < 10:
                return torch.randn(self.batch_size, 1, 28, 28), torch.randint(0, 9, (self.batch_size,))
            raise StopIteration


    train_loader = RandData(batch_size)
    test_loader = RandData(batch_size)

    class Net(nn.Module):
        """ Basic conv net.
        """

        def __init__(self):
            super().__init__()
            # Testing several different activations
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.Tanh(),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.ConvTranspose2d(20, 20, 1),
                nn.AdaptiveAvgPool2d(output_size=(4, 4)),
                nn.Softplus(),
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(320, 50),
                nn.BatchNorm1d(50),
                nn.ReLU(),
                nn.Linear(50, 10),
                nn.ELU(),
                nn.Softmax(dim=1)
            )

        def forward(self, x):
            """ Run the model.
            """
            x = self.conv_layers(x)
            x = x.view(-1, 320)
            x = self.fc_layers(x)
            return x

    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    def train(model, device, train_loader, optimizer, _, cutoff=20):
        model.train()
        num_examples = 0
        for _, (data, target) in enumerate(train_loader):
            num_examples += target.shape[0]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, torch.eye(10)[target])
            # loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            # if batch_idx % 10 == 0:
            #     # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #         # 100. * batch_idx / len(train_loader), loss.item()))
            if num_examples > cutoff:
                break

    device = torch.device('cpu')
    train(model, device, train_loader, optimizer, 1)

    next_x, _ = next(iter(train_loader))
    np.random.seed(0)
    inds = np.random.choice(next_x.shape[0], 3, replace=False)
    #e = shap.DeepExplainer(model, next_x[inds, :, :, :])
    explain_obj = SHAPExplainer([model],next_x[inds, :, :, :] , domain="deep")

    test_x, _ = next(iter(test_loader))
    input_tensor = test_x[:1]
    input_tensor.requires_grad = True
    # shap_values = e.shap_values(input_tensor)
    shap_values = explain_obj.get_explaination(input_tensor)
    print("Testing Completed with the explanation : ", shap_values[0].shape)
    #print(shap_values)




if __name__ == '__main__':
    test_shap_explainer()