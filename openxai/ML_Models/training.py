import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision import transforms
import copy
import ipdb
from sklearn.metrics import f1_score, accuracy_score

from openxai.ML_Models.LR.model import LogisticRegression
import openxai.ML_Models.ANN.model as model_ann
import openxai.ML_Models.data_loader as loader

def training(model, train_loader, test_loader, ml_model, dir_name, learning_rate, epochs, dataset,
             adv_train_params=None):

    loaders = {'train': train_loader,
               'test': test_loader}

    # model collector
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # declaring optimizer and loss
    if dataset == ('mnist' or 'cifar10'):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        if ml_model == 'svm':
            criterion = SVM_Loss()
        else:
            criterion = nn.BCELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    # training
    for e in range(epochs):
        print('Epoch {}/{}'.format(e, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_acc = 0.0
            running_f1 = 0.0

            if dataset == 'gaussian':
                for i, (inputs, labels, weights, masks, masked_weights, probs, cluster_idx) in enumerate(loaders[phase]):

                    inputs = inputs.to(device)
                    labels = labels.to(device).long()
                    labels_one_hot = torch.stack([1-labels, labels], dim=1)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs.float())
                        if ml_model=='svm':
                            loss = criterion(outputs, 2*(labels.float()-0.5))
                        else:
                            loss = criterion(outputs, labels_one_hot.float())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # print(outputs)

                    # statistics
                    # CHIRAG: calculate accuracy for SVM
                    preds = outputs.data[:, 1] >= 0.5
                    # running_acc += (preds.view(-1).long() == labels).sum()/labels.shape[0]
                    # print('true labels', labels.numpy())
                    # print('predicted labels', preds.view(-1).long().numpy())
                    running_acc += accuracy_score(labels.numpy(), preds.view(-1).long().numpy())
                    running_loss += loss.item()  #  * inputs.size(0)
                    running_f1 += f1_score(labels.numpy(), preds.view(-1).long().numpy())


            else:

                for i, (inputs, labels) in enumerate(loaders[phase]):

                    inputs = inputs.to(device)
                    labels = labels.to(device).type(torch.long)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        y_pred = model(inputs.float())

                        if ml_model=='svm':
                            loss = criterion(y_pred, 2*(labels.float()-0.5))
                        elif ml_model == 'lr':
                            loss = criterion(y_pred[:, 1].float(), labels.float())
                        elif ml_model=='ann':
                            loss = criterion(y_pred, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    preds = y_pred.data[:, 1] >= 0.5
                    # print('probs', y_pred)
                    # print('true labels', labels.numpy())
                    # print('predicted labels', preds.view(-1).long().numpy())
                    running_acc += accuracy_score(labels.numpy(), preds.view(-1).long().numpy())
                    running_loss += loss.item()  #  * inputs.size(0)
                    running_f1 += f1_score(labels.numpy(), preds.view(-1).long().numpy())


            epoch_loss = running_loss / (i+1)
            epoch_acc = running_acc / (i+1)
            epoch_f1 = running_f1 / (i+1)

            print(f'{phase}: Loss: {epoch_loss:.4f} | F1-score: {epoch_f1:.4f} | Accuracy: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    if adv_train_params['type'] == 'none':
        # save vanilla model
        torch.save(model.state_dict(best_model_wts),
                   './ML_Models/Saved_Models/{}/{}_{}_{}_acc_{:.2f}.pt'.format(dir_name, dataset, ml_model, learning_rate, best_acc))

def main():

    style = 'none'
    epochs = 50
    names = ['adult', 'gaussian', 'compas']
    ml_model = 'lr'
    dir_name = 'LR'

    for name in names:

        adv_training_params = {'type': 'none',       # {'gaussian', 'none', or 'attack'}
                               'parameter': None}    # {std, epsilon, None}

        if name == 'compas':
            dataset_train = loader.DataLoader_Tabular(path='./Data_Sets/COMPAS/',
                                              filename='compas-scores-train.csv', label='risk')

            dataset_test = loader.DataLoader_Tabular(path='./Data_Sets/COMPAS/',
                                             filename='compas-scores-test.csv', label='risk')

            input_size = dataset_train.get_number_of_features()

            # Define the model
            if ml_model == 'ann':
                model = model_ann.ANN(input_size, 18, 9, 3, 2)
            elif ml_model == 'svm':
                model = SVM(input_size, num_of_classes=2)
            elif ml_model == 'lr':
                model = LogisticRegression(input_size, num_of_classes = 2)
            
            else:
                print('Invalid model type')
                exit(0)

            # COMPAS training
            trainloader = DataLoader(dataset_train, batch_size=32, shuffle=True)
            testloader = DataLoader(dataset_test, batch_size=32, shuffle=False)

            training(model, trainloader, testloader, ml_model, dir_name, 0.002, epochs, 'compas', adv_training_params)

        elif name == 'adult':
            dataset_train = loader.DataLoader_Tabular(path='./Data_Sets/Adult/',
                                              filename='adult-train.csv', label='income')

            dataset_test = loader.DataLoader_Tabular(path='./Data_Sets/Adult/',
                                             filename='adult-test.csv', label='income')

            input_size = dataset_train.get_number_of_features()

            # Define the model
            if ml_model == 'ann':
                model = model_ann.ANN(input_size, 18, 9, 3, 2)
            elif ml_model == 'svm':
                model = SVM(input_size, num_of_classes=2)
            elif ml_model == 'lr':
                model = LogisticRegression(input_size, num_of_classes = 2)
            else:
                print('Invalid model type')
                exit(0)

            # Adult training
            trainloader = DataLoader(dataset_train, batch_size=256, shuffle=True)
            testloader = DataLoader(dataset_test, batch_size=256, shuffle=False)

            # Train the model
            training(model, trainloader, testloader, ml_model, dir_name, 0.002, epochs, 'adult', adv_training_params)

        elif name == 'german':
            dataset_train = loader.DataLoader_Tabular(path='./Data_Sets/German_Credit_Data/',
                                              filename='german-train.csv', label='credit-risk')

            dataset_test = loader.DataLoader_Tabular(path='./Data_Sets/German_Credit_Data/',
                                             filename='german-test.csv', label='credit-risk')
            input_size = dataset_train.get_number_of_features()

            # Define the model
            if ml_model == 'ann':
                model = model_ann.ANN_softmax(input_layer=input_size, hidden_layer_1=100, num_of_classes=2)

            elif ml_model == 'svm':
                model = SVM(input_size, num_of_classes=2)
            elif ml_model == 'lr':
                model = LogisticRegression(input_size, num_of_classes = 2)
            else:
                print('Invalid model type')
                exit(0)

            # German training
            trainloader = DataLoader(dataset_train, batch_size=32, shuffle=True)
            testloader = DataLoader(dataset_test, batch_size=21, shuffle=True)

            # Train the model
            training(model, trainloader, testloader, ml_model, dir_name, 0.002, epochs, 'german', adv_training_params)

        elif name == 'gaussian':
            dataset_train = loader.DataLoader_Tabular(path='gaussian',
                                                      filename='train', label='y')

            dataset_test = loader.DataLoader_Tabular(path='gaussian',
                                                     filename='test', label='y')
            input_size = dataset_train.get_number_of_features()

            # Define the model
            if ml_model == 'ann':
                model = model_ann.ANN_softmax(input_size, hidden_layer_1=100, num_of_classes=2)
            elif ml_model == 'svm':
                model = SVM(input_size, num_of_classes=2)
            elif ml_model == 'lr':
                model = LogisticRegression(input_size, num_of_classes = 2)
            else:
                print('Invalid model type')
                exit(0)


            # Gaussian training
            trainloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
            testloader = DataLoader(dataset_test, batch_size=64, shuffle=True)

            # Train the model
            training(model, trainloader, testloader, ml_model, dir_name, 0.002, epochs, dataset='gaussian', adv_train_params=adv_training_params)

        elif name == 'mnist':
            img_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = MNIST(root='./Data_Sets/', train=True, download=True, transform=img_transform)
            dataset_test = MNIST(root='./Data_Sets/', train=False, download=True, transform=img_transform)

            # Define the model
            model = model_ann.Conv_Net(n_channels=1, image_size=28, kernel_size=3)

            trainloader = DataLoader(dataset_train, batch_size=128, shuffle=True)
            testloader = DataLoader(dataset_test, batch_size=128, shuffle=True)

            # Train the model
            training(model, trainloader, testloader, 0.002, epochs, 'mnist', adv_training_params)

        elif name == 'cifar10':
            img_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = CIFAR10(root='./Data_Sets/CIFAR10', train=True, download=True, transform=img_transform)
            dataset_test = CIFAR10(root='./Data_Sets/CIFAR10', train=False, download=True, transform=img_transform)


            trainloader = DataLoader(dataset_train, batch_size=256, shuffle=True)
            testloader = DataLoader(dataset_test, batch_size=256, shuffle=True)

            # Define the model
            model = model_ann.Conv_Net(n_channels=3, image_size=32, kernel_size=5)

            # Train the model
            training(model, trainloader, testloader, 0.02, epochs, 'cifar10', adv_training_params)

if __name__ == "__main__":
    # execute training
    main()
