import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import requests

activation_functions = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(),
                        'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}

dataverse_prefix = 'https://dataverse.harvard.edu/api/access/datafile/'
dataverse_ids = {
    'lr': {
        'synthetic': '6718576', 'adult': '6718044', 'compas': '6718042', 'german': '6718043',
        'heloc': '6718046', 'rcdv': '7093736', 'lending-club': '6990766', 'student': '7093732',
    },
    'ann': {
        'synthetic': '6718575', 'adult': '6718041', 'compas': '6718040', 'german': '6718047',
        'heloc': '6718045', 'rcdv': '7093738', 'lending-club': '6990764', 'student': '7093735'
    }
}

def LoadModel(data_name: str, ml_model, pretrained: bool = True):
    """
    Load a pretrained model
    :param data_name: string with name of dataset
    :param ml_model: string with name of model; 'lr' or 'ann'
    :param pretrained: boolean, whether to load a pretrained model
    :return: model
    """
    if pretrained:
        os.makedirs('./pretrained', exist_ok=True)
        if data_name in ['synthetic', 'adult', 'compas', 'german', 'heloc', 'rcdv', 'lending-club', 'student']:
            r = requests.get(dataverse_prefix + dataverse_ids[ml_model][data_name], allow_redirects=True)
            model_path = f'./pretrained/{ml_model}_{data_name}.pt'
            open(model_path, 'wb').write(r.content)
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            num_features = next(iter(state_dict.values()))[1]
            if ml_model == 'ann':
                model = ArtificialNeuralNetwork(num_features, [100,100], n_class=2)
            elif ml_model == 'lr':
                model = LogisticRegression(num_features)
            model.load_state_dict(state_dict)
        else:
            raise NotImplementedError(
                 'The current version of >LoadModel< does not support this data set.')
    else:
        raise NotImplementedError(
             'The current version of >LoadModel< does not support training a ML model from scratch, yet.')
    return model


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, n_class = 2):
        '''
        Initializes the logistic regression model
        :param input_dim: int, number of features
        :param n_class: int, number of classes
        '''
        super().__init__()
        self.input_dim = input_dim
        self.n_class = n_class
        self.linear = nn.Linear(self.input_dim, self.n_class)
        self.name = 'LogisticRegression'
        self.abbrv = 'lr'
        
    def return_ground_truth_importance(self):
        return self.linear.weight[1, :] - self.linear.weight[0, :]
        
    def forward(self, x):
        return F.softmax(self.linear(x), dim=-1)
    
    def predict(self, data):
        """
        Predict method required for CFE-Models
        :param data: torch or list
        :return: numpy array of predictions
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = torch.squeeze(data)
            
        output = self.forward(input).detach().numpy()

        return output

class ArtificialNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers, n_class = 2, activation = 'relu'):
        """
        Initializes the artificial neural network model
        :param input_dim: int, number of features
        :param hidden_layers: list of int, number of neurons in each hidden layer
        :param n_class: int, number of classes
        """
        super().__init__()
        
        # Construct layers
        model_layers = []
        previous_layer = input_dim
        for layer in hidden_layers:
            model_layers.append(nn.Linear(previous_layer, layer))
            model_layers.append(activation_functions[activation])
            previous_layer = layer
        model_layers.append(nn.Linear(previous_layer, n_class))
        self.network = nn.Sequential(*model_layers)
        self.name = 'ArtificialNeuralNetwork'
        self.abbrv = 'ann'
    
    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)
    
    def predict_with_logits(self, x):
        return self.network(x)
    
    def predict_proba(self, data):
        # Currently used by SHAP
        input = data if torch.is_tensor(data) else torch.from_numpy(np.array(data))
        return self.forward(input.float()).detach().numpy()
    
    def predict(self, data):
        # Currently used by LIME
        input = torch.squeeze(data) if torch.is_tensor(data) else torch.from_numpy(np.array(data))
        return self.forward(input.float()).detach().numpy()
    
    def L_relu(self, data):
        # Deprecated: used in stability experiments
        return F.relu(self.input1(data))
