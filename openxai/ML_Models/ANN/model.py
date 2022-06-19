import torch
import numpy as np
from typing import Union

from torch import nn
import torch.nn.functional as F


class ANN_softmax(nn.Module):
    def __init__(self, input_layer: int, hidden_layer_1: int, num_of_classes: int = 1):
        """
        Defines the structure of the neural network
        :param input_layer: int > 0, number of neurons for this layer
        :param hidden_layer_1: int > 0, number of neurons for this layer
        :param hidden_layer_2: int > 0, number of neurons for this layer
        :param output_layer: int > 0, number of neurons for this layer
        :param num_of_classes: int > 0, number of classes
        """
        super().__init__()
        
        # Layer
        self.input1 = nn.Linear(input_layer, hidden_layer_1)
        self.input2 = nn.Linear(hidden_layer_1, num_of_classes)
        
        # Activation
        # self.relu = nn.ReLU()
        # self.softmax = nn.functional.softmax(dim=1)
    
    def forward(self, x: torch.FloatTensor):
        """
        Forwardpass through the network
        :param input: tabular data
        :return: prediction
        """
        return F.softmax(self.input2(F.relu(self.input1(x))), dim=1)
    
    def predict_with_logits(self, x: torch.FloatTensor):
        """
        Forwardpass through the network
        :param input: tabular data
        :return: prediction
        """
        
        output = self.input1(x)
        output = F.relu(output)
        output = self.input2(output)
        
        return output
    
    def predict_proba(self, data: Union[torch.FloatTensor, np.array]) -> np.array:
        """
        Computes probabilistic output for c classes
        :param data: torch tabular input
        :return: np.array
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = data.float()

        return self.forward(input).detach().numpy()
    
    def predict(self, data):
        """
        :param data: torch or list
        :return: np.array with prediction
        """
        
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = torch.squeeze(data).float()
        
        return self.forward(input).detach().numpy()
    
    def L_relu(self, data):
        
        output = self.input1(data)
        output = F.relu(output)
        
        return output
