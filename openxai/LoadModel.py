import torch
# models
import ML_Models.ANN.model as model_ann
from ML_Models.LR.model import LogisticRegression
# (Train & Test) Loaders
import dataloader as loaders
import subprocess
import os.path


def LoadModel(data_name: str, ml_model, pretrained: bool = True):

    # obtain inputs to infer number of features
    if data_name == 'synthetic':
        # this has to be fixed for the moment as the ANN model is trained on this data
        gauss_params = {
            'n_samples': 1000,
            'dim': 20,
            'n_clusters': 10,
            'distance_to_center': 6,
            'test_size': 0.25,
            'upper_weight': 1,
            'lower_weight': -1,
            'seed': 564,
            'sigma': None,
            'sparsity': 0.25
        }
        loader_train, loader_test = loaders.return_loaders(data_name=data_name,
                                                           download=False,
                                                           batch_size=32,  # arbitrary
                                                           gauss_params=gauss_params)
        data_iter = iter(loader_test)
        inputs, labels, weights, masks, masked_weights, probs, cluster_idx = data_iter.next()
    else:
        loader_train, loader_test = loaders.return_loaders(data_name=data_name,
                                                           download=True,
                                                           batch_size=32)  # arbitrary
        data_iter = iter(loader_test)
        inputs, labels = data_iter.next()

    if pretrained:
        if data_name == 'synthetic':
            if ml_model == 'ann':
                model_name = 'gaussian_lr_0.002_acc_0.91.pt'
                url = 'https://github.com/AI4LIFE-GROUP/OpenXAI/raw/main/openxai/ML_Models/Saved_Models/ANN/' + model_name
                out = './openxai/ML_Models/Saved_Models/ANN/'
                model_path = out + model_name
                if not os.path.exists(model_path):
                    subprocess.run(["wget", "-P", out, url], check=True)
                model = model_ann.ANN_softmax(input_layer=inputs.shape[1],
                                              hidden_layer_1=100,
                                              num_of_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

            elif ml_model == 'lr':
                model_name = 'gaussian_lr_0.002_acc_0.73.pt'
                url = 'https://github.com/AI4LIFE-GROUP/OpenXAI/raw/main/openxai/ML_Models/Saved_Models/LR/' + model_name
                out = './openxai/ML_Models/Saved_Models/LR/'
                model_path = out + model_name
                if not os.path.exists(model_path):
                    subprocess.run(["wget", "-P", out, url], check=True)
                model = LogisticRegression(input_dim=inputs.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        elif data_name == 'adult':
            if ml_model == 'ann':
                model_name = 'adult_lr_0.002_acc_0.83.pt'
                url = 'https://github.com/AI4LIFE-GROUP/OpenXAI/raw/main/openxai/ML_Models/Saved_Models/ANN/' + model_name
                out = './openxai/ML_Models/Saved_Models/ANN/'
                model_path = out + model_name
                if not os.path.exists(model_path):
                    subprocess.run(["wget", "-P", out, url], check=True)
                model = model_ann.ANN_softmax(input_layer=inputs.shape[1],
                                              hidden_layer_1=100,
                                              num_of_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

            elif ml_model == 'lr':
                model_name = 'adult_lr_0.002_acc_0.84.pt'
                url = 'https://github.com/AI4LIFE-GROUP/OpenXAI/raw/main/openxai/ML_Models/Saved_Models/LR/' + model_name
                out = './openxai/ML_Models/Saved_Models/LR/'
                model_path = out + model_name
                if not os.path.exists(model_path):
                    subprocess.run(["wget", "-P", out, url], check=True)

                model = LogisticRegression(input_dim=inputs.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        elif data_name == 'compas':
            if ml_model == 'ann':
                model_name = 'compas_lr_0.002_acc_0.85.pt'
                url = 'https://github.com/AI4LIFE-GROUP/OpenXAI/raw/main/openxai/ML_Models/Saved_Models/ANN/' + model_name
                out = './openxai/ML_Models/Saved_Models/ANN/'
                model_path = out + model_name
                if not os.path.exists(model_path):
                    subprocess.run(["wget", "-P", out, url], check=True)
                model = model_ann.ANN_softmax(input_layer=inputs.shape[1],
                                              hidden_layer_1=100,
                                              num_of_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

            elif ml_model == 'lr':
                model_name = 'compas_lr_0.002_acc_0.85.pt'
                url = 'https://github.com/AI4LIFE-GROUP/OpenXAI/raw/main/openxai/ML_Models/Saved_Models/LR/' + model_name
                out = './openxai/ML_Models/Saved_Models/LR/'
                model_path = out + model_name
                if not os.path.exists(model_path):
                    subprocess.run(["wget", "-P", out, url], check=True)

                model = LogisticRegression(input_dim=inputs.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        elif data_name == 'german':
            if ml_model == 'ann':
                model_name = 'german_lr_0.002_acc_0.71.pt'
                url = 'https://github.com/AI4LIFE-GROUP/OpenXAI/raw/main/openxai/ML_Models/Saved_Models/ANN/' + model_name
                out = './openxai/ML_Models/Saved_Models/ANN/'
                model_path = out + model_name
                if not os.path.exists(model_path):
                    subprocess.run(["wget", "-P", out, url], check=True)
                model = model_ann.ANN_softmax(input_layer=inputs.shape[1],
                                              hidden_layer_1=100,
                                              num_of_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

            elif ml_model == 'lr':
                model_name = 'german_lr_0.002_acc_0.72.pt'
                url = 'https://github.com/AI4LIFE-GROUP/OpenXAI/raw/main/openxai/ML_Models/Saved_Models/LR/' + model_name
                out = './openxai/ML_Models/Saved_Models/LR/'
                model_path = out + model_name
                if not os.path.exists(model_path):
                    subprocess.run(["wget", "-P", out, url], check=True)
                model = LogisticRegression(input_dim=inputs.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        elif data_name == 'heloc':
            if ml_model == 'ann':
                model_name = 'heloc_lr_0.002_acc_0.74.pt'
                url = 'https://github.com/AI4LIFE-GROUP/OpenXAI/raw/main/openxai/ML_Models/Saved_Models/ANN/' + model_name
                out = './openxai/ML_Models/Saved_Models/ANN/'
                model_path = out + model_name
                if not os.path.exists(model_path):
                    subprocess.run(["wget", "-P", out, url], check=True)
                model = model_ann.ANN_softmax(input_layer=inputs.shape[1],
                                              hidden_layer_1=100,
                                              num_of_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

            elif ml_model == 'lr':
                model_name = 'heloc_lr_0.002_acc_0.72.pt'
                url = 'https://github.com/AI4LIFE-GROUP/OpenXAI/raw/main/openxai/ML_Models/Saved_Models/LR/' + model_name
                out = './openxai/ML_Models/Saved_Models/LR/'
                model_path = out + model_name
                if not os.path.exists(model_path):
                    subprocess.run(["wget", "-P", out, url], check=True)
                model = LogisticRegression(input_dim=inputs.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            raise NotImplementedError(
                 'The current version of >LoadModel< does not support this data set.')

    else:
        raise NotImplementedError(
             'The current version of >LoadModel< does not support training a ML model from scratch, yet.')

    return model
