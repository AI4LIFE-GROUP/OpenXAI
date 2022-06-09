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
        return self.input1(x)       


class SVM_Loss(nn.Module):    
    def __init__(self):
        super(SVM_Loss,self).__init__()

    def forward(self, outputs, labels):
        return torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))/outputs.shape[0]
