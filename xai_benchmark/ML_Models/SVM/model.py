import torch
import numpy as np
from typing import Union

from torch import nn
import torch.nn.functional as F


class SVM(nn.Module):
    def __init__(self, input_layer, num_of_classes=1):
        """
        Defines the structure of the neural network
        :param input_layer: int > 0, number of neurons for this layer
        :param output_layer: int > 0, number of neurons for this layer
        :param num_of_classes: int > 0, number of classes
        """
        super().__init__()

        # Layer
        self.input1 = nn.Linear(input_layer, num_of_classes)


    def forward(self, x):
        """
        Forwardpass through the network
        :param input: tabular data
        :return: prediction
        """
        prb_1 = self.input1(x)
        prb_0 = 1 - prb_1
        
        return torch.concat((prb_0, prb_1), dim = 1)
        


    def predict(self, data):
        """
        predict method for CFE-Models which need this method.
        :param data: torch or list
        :return: np.array with prediction
        """
        
        output = np.zeros((data.shape[0], 2))
        
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
            # input = torch.squeeze(input)
        else:
            input = torch.squeeze(data)
            
        output[:, 0] = 1 - self.forward(input).detach().numpy().reshape(-1)
        output[:, 1] = self.forward(input).detach().numpy().reshape(-1)

        return output


class SVM_Loss(nn.Module):    
    def __init__(self):
        super(SVM_Loss,self).__init__()

    def forward(self, outputs, labels):
        prb_1_outputs = outputs[:, 1]
        return torch.sum(torch.clamp(1 - prb_1_outputs.t()*labels, min=0))/outputs.shape[0]
