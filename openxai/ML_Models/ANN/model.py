import torch
import numpy as np
from typing import Union

from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_channels, image_size, kernel_size, num_of_classes=10):
        super().__init__()
        
        # formula for output dim: [(input dim + 2 padding - kernel)/(stride)] +1
        # input_dim=image_size (in the beginning), padding=0, stride=1;
        # after pooling layer, we divide output_dim by pooling_dim (here fixed to 2)
        
        self.final_multiplier = ((image_size + 0 - kernel_size)/1 + 1)/2
        self.final_multiplier = int(((self.final_multiplier + 0 - kernel_size)/1 + 1)/2)
        
        self.conv1 = nn.Conv2d(n_channels, 6, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=kernel_size)
        self.fc1 = nn.Linear(16 * self.final_multiplier * self.final_multiplier, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_of_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x)
        return x
    
    def predict_proba(self, data):
        """
        Computes probabilistic output for c classes
        :param data: torch tabular input
        :return: np.array
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
            # input = torch.squeeze(input)
        else:
            input = data.float()

        return self.forward(input).detach().numpy()


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


class ANN_sigmoid(nn.Module):
    def __init__(self, input_layer, hidden_layer_1, num_of_classes=1):
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
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        Forwardpass through the network
        :param input: tabular data
        :return: prediction
        """
        output = self.input1(x)
        output = self.relu(output)
        output = self.input2(output)
        #output = self.sigmoid(output)
        
        return output
    
    def predict_with_logits(self, x):
        """
        Forwardpass through the network
        :param input: tabular data
        :return: prediction
        """
        output = self.input1(x)
        output = self.relu(output)
        output = self.input2(output)
        
        return output
    
    def predict_proba(self, data):
        """
        Computes probabilistic output for two classes
        :param data: torch tabular input
        :return: np.array
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
            # input = torch.squeeze(input)
        else:
            input = torch.squeeze(data)

        return self.forward(input).detach().numpy().squeeze()

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

