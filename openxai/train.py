import torch
from torch import nn
import copy
from sklearn.metrics import f1_score, accuracy_score
from openxai.dataloader import return_loaders
from openxai.model import LogisticRegression
from openxai.model import ArtificialNeuralNetwork
from openxai.experiment_utils import print_summary

def train_model(ml_model, dataset, learning_rate, epochs, batch_size, scaler='minmax', seed=0,
             pos_class_weight=0.5, mean_prediction_bound=1.0, warmup=5, verbose=False):
    """
    Train a (binary classificaiton) model
    :param ml_model: string with abbreviated name of model; 'lr' or 'ann'
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
    trainloader, testloader = return_loaders(dataset, download=False,
                                             batch_size=batch_size, scaler=scaler)
    input_size = trainloader.dataset.data.shape[-1]
    loaders = {'train': trainloader, 'test': testloader}

    # Define the model
    torch.manual_seed(seed)
    if ml_model == 'ann':
        model = ArtificialNeuralNetwork(input_size, [100, 100], n_class=2)
    elif ml_model == 'lr':
        model = LogisticRegression(input_size, n_class = 2)
    else:
        print('Invalid model type')
        exit(0)

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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