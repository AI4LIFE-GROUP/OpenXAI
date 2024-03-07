import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import copy
import requests
from sklearn.metrics import f1_score, accuracy_score
from openxai.dataloader import ReturnLoaders
from openxai.experiment_utils import print_summary

activation_functions = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(),
                        'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}

dataverse_prefix = 'https://dataverse.harvard.edu/api/access/datafile/'
dataverse_ids = {
    'lr': {
        'adult': '8550955', 'compas': '8550949', 'gaussian': '8550960', 'german': '8550945',
        'gmsc': '8550948', 'heart': '8550956', 'heloc': '8550950', 'pima': '8550959'
    },
    'ann': {
        'adult': '8550958', 'compas': '8550951', 'gaussian': '8550957', 'german': '8550946',
        'gmsc': '8550947', 'heart': '8550954', 'heloc': '8550952', 'pima': '8550953'
    },
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
        model_path = './models/pretrained/'
        os.makedirs(model_path, exist_ok=True)
        if data_name in dataverse_ids[ml_model]:
            r = requests.get(dataverse_prefix + dataverse_ids[ml_model][data_name], allow_redirects=True)
            model_filename = f'{ml_model}_{data_name}.pt'
            open(model_path+model_filename, 'wb').write(r.content)
            state_dict = torch.load(model_path+model_filename, map_location=torch.device('cpu'))
            num_features = next(iter(state_dict.values())).shape[1]
            if ml_model == 'ann':
                model = ArtificialNeuralNetwork(num_features, [100, 100])
            elif ml_model == 'lr':
                model = LogisticRegression(num_features)
            model.load_state_dict(state_dict)
        else:
            raise NotImplementedError(
                f'The current version of >LoadModel< does not support this data set for {ml_model.upper()} models.')
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
        self.name = 'LogisticRegression'
        self.abbrv = 'lr'

        # Construct layers
        self.input_dim = input_dim
        self.n_class = n_class
        self.linear = nn.Linear(self.input_dim, self.n_class)
        
    def return_ground_truth_importance(self):
        return self.linear.weight[1, :] - self.linear.weight[0, :]
        
    def forward(self, x):
        return F.softmax(self.linear(x), dim=-1)
    
    def predict_with_logits(self, x):
        return self.linear(x)
    
    def predict(self, data, argmax=False):
        """
        Predict method required for CFE-Models
        :param data: torch or list
        :return: numpy array of predictions
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data))
        else:
            input = torch.squeeze(data)
            
        output = self.forward(input.float()).detach().numpy()

        return output.argmax(axis=-1) if argmax else output

class ArtificialNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers, n_class = 2, activation = 'relu'):
        """
        Initializes the artificial neural network model
        :param input_dim: int, number of features
        :param hidden_layers: list of int, number of neurons in each hidden layer
        :param n_class: int, number of classes
        """
        super().__init__()
        self.name = 'ArtificialNeuralNetwork'
        self.abbrv = 'ann'
        
        # Construct layers
        model_layers = []
        previous_layer = input_dim
        for layer in hidden_layers:
            model_layers.append(nn.Linear(previous_layer, layer))
            model_layers.append(activation_functions[activation])
            previous_layer = layer
        model_layers.append(nn.Linear(previous_layer, n_class))
        self.network = nn.Sequential(*model_layers)
    
    def predict_layer(self, x, hidden_layer_idx=0, post_act=True):
        """
        Returns the representation of the input tensor at the specified layer
        :param x: torch.tensor, input tensor
        :param layer: int, layer number
        :param post_act: bool, whether to return the activations before or after the activation function
        """
        if hidden_layer_idx >= len(self.network) // 2:
            raise ValueError(f'The model has only {len(self.network) // 2} hidden layers, but hidden layer {hidden_layer_idx} was requested (indexing starts at 0).')
        
        network_idx = 2 * hidden_layer_idx + int(post_act)
        return self.network[:network_idx+1](x)
    
    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)
    
    def predict_with_logits(self, x):
        return self.network(x)
    
    def predict_proba(self, x):
        # Currently used by SHAP
        input = x if torch.is_tensor(x) else torch.from_numpy(np.array(x))
        return self.forward(input.float()).detach().numpy()
    
    def predict(self, x, argmax=False):
        # Currently used by LIME
        input = torch.squeeze(x) if torch.is_tensor(x) else torch.from_numpy(np.array(x))
        output = self.forward(input.float()).detach().numpy()
        return output.argmax(axis=-1) if argmax else output

def train_model(model_name, dataset, learning_rate, epochs, batch_size, scaler='minmax', seed=0,
                pos_class_weight=0.5, mean_prediction_bound=1.0, warmup=5, verbose=False):
    """
    Train a (binary classificaiton) model
    :param model_name: string with abbreviated name of model; 'lr' or 'ann'
    :param dataset: string with name of dataset
    :param learning_rate: float, learning rate
    :param epochs: int, number of epochs
    :param batch_size: int, batch size
    :param scaler: string, type of scaler to use; 'minmax', 'standard', or 'none'
    :param seed: int, random seed to initialize model
    :param pos_class_weight: float, weight for positive class in loss function
    :param mean_prediction_bound: float, bound on the mean prediction (avoids predicting all 0s or 1s)
    :param warmup: int, number of epochs before starting to track best model
    :param verbose: boolean, whether to print training progress
    :return: trained model, best accuracy, best epoch
    """
    # Dataloaders
    file_path = f'./data/{dataset}/{dataset}'
    all_splits_downloaded = os.path.exists(file_path+'-train.csv') and os.path.exists(file_path+'-test.csv')
    download = False if all_splits_downloaded else True
    trainloader, testloader = ReturnLoaders(dataset, download, batch_size, scaler)
    input_size = trainloader.dataset.data.shape[-1]
    loaders = {'train': trainloader, 'test': testloader}

    # Define the model
    torch.manual_seed(seed)
    if model_name == 'ann':
        model = ArtificialNeuralNetwork(input_size, [100, 100], n_class=2)
    elif model_name == 'lr':
        model = LogisticRegression(input_size, n_class = 2)
    else:
        print('Invalid model type')
        exit(0)

    # Current version uses CPU
    device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    class_weights = torch.FloatTensor([1-pos_class_weight, pos_class_weight])
    criterion = nn.CrossEntropyLoss(weight = class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize trackers
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, best_epoch = 0, 0

    # Training loop
    for e in range(epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_acc, running_f1, n_inputs = 0.0, 0.0, 0.0, 0
            for inputs, labels in loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).type(torch.long)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(inputs.float())
                    loss = criterion(y_pred, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Track statistics
                preds = y_pred.data[:, 1] >= 0.5
                running_acc += accuracy_score(labels.numpy(), preds.view(-1).long().numpy()) * inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_f1 += f1_score(labels.numpy(), preds.view(-1).long().numpy(), zero_division=0) * inputs.size(0)
                n_inputs += inputs.size(0)

            epoch_loss = running_loss / n_inputs
            epoch_acc = running_acc / n_inputs
            epoch_f1 = running_f1 / n_inputs

            if verbose:
                print('Epoch {}/{}'.format(e, epochs - 1))
                print('-' * 10)
                print(f'{phase}: Loss: {epoch_loss:.4f} | F1-score: {epoch_f1:.4f} | Accuracy: {epoch_acc:.4f}')
            
            X_test = torch.FloatTensor(loaders['test'].dataset.data)
            mean_pred = (model(X_test)[:, 1] >= 0.5).to(int).detach().numpy().mean()
    
            if (phase == 'test') and (epoch_acc > best_acc) and (e > warmup):
                pos_bound = 1 if mean_prediction_bound > 0.5 else -1
                if pos_bound * (mean_prediction_bound - mean_pred) > 0:
                    best_epoch, best_acc, best_model_wts = e, epoch_acc, copy.deepcopy(model.state_dict())
                    print(e, round(epoch_acc*100, 2), f"Best Seen Test Acc (Mean Pred = {round(mean_pred, 2)})")

    # No best epoch found
    if best_epoch == 0:
        print('No epoch found within prediction bounds, using last epoch.')
        best_epoch, best_acc, best_model_wts = e, epoch_acc, copy.deepcopy(model.state_dict())

    # Load best weights
    model.load_state_dict(best_model_wts)
    print_summary(model, trainloader, testloader)

    return model, best_acc, best_epoch