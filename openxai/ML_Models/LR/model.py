import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_of_classes = 2):
        ''' Initializes the LogisticRegression.
        '''
        
        super().__init__()
        
        self.input_dim = input_dim
        self.num_of_classes = num_of_classes
        
        self.linear = nn.Linear(self.input_dim, self.num_of_classes)
        # self.softmax = nn.Softmax(dim = 1)
        
    def return_ground_truth_importance(self, x):
        """ Returns a vector containing the ground truth feature attributions for input x.
        """
        # the true feature attribution is the same for all points x
        return self.linear.weight[1, :] - self.linear.weight[0, :]
        
    def forward(self, x):
        return F.softmax(self.linear(x), dim=1)
    
    def predict(self, data):
        """
        predict method for CFE-Models which need this method.
        :param data: torch or list
        :return: np.array with prediction
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = torch.squeeze(data)
            
        output = self.forward(input).detach().numpy()

        return output


