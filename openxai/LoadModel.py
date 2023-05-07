import os
import torch
import requests

# models
import openxai.ML_Models.ANN.model as model_ann
from openxai.ML_Models.LR.model import LogisticRegression
# (Train & Test) Loaders
import openxai.dataloader as loaders


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
        os.makedirs('./pretrained', exist_ok=True)
        if data_name == 'synthetic':
            if ml_model == 'ann':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/6718575', allow_redirects=True)
                model_path = './pretrained/ann_synthetic.pt'
                open(model_path, 'wb').write(r.content)
                model = model_ann.ANN_softmax(input_layer=inputs.shape[1],
                                              hidden_layer_1=100,
                                              num_of_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                
            elif ml_model == 'lr':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/6718576', allow_redirects=True)
                model_path = './pretrained/lr_synthetic.pt'
                open(model_path, 'wb').write(r.content)
                model = LogisticRegression(input_dim=inputs.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        elif data_name == 'adult':
            if ml_model == 'ann':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/6718041', allow_redirects=True)
                model_path = './pretrained/ann_adult.pt'
                open(model_path, 'wb').write(r.content)
                model = model_ann.ANN_softmax(input_layer=inputs.shape[1],
                                              hidden_layer_1=100,
                                              num_of_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                
            elif ml_model == 'lr':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/6718044', allow_redirects=True)
                model_path = './pretrained/lr_adult.pt'
                open(model_path, 'wb').write(r.content)
                model = LogisticRegression(input_dim=inputs.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                
        elif data_name == 'compas':
            if ml_model == 'ann':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/6718040', allow_redirects=True)
                model_path = './pretrained/ann_compas.pt'
                open(model_path, 'wb').write(r.content)
                model = model_ann.ANN_softmax(input_layer=inputs.shape[1],
                                              hidden_layer_1=100,
                                              num_of_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                
            elif ml_model == 'lr':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/6718042', allow_redirects=True)
                model_path = './pretrained/lr_compas.pt'
                open(model_path, 'wb').write(r.content)
                model = LogisticRegression(input_dim=inputs.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                
        elif data_name == 'german':
            if ml_model == 'ann':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/6718047', allow_redirects=True)
                model_path = './pretrained/ann_german.pt'
                open(model_path, 'wb').write(r.content)
                model = model_ann.ANN_softmax(input_layer=inputs.shape[1],
                                              hidden_layer_1=100,
                                              num_of_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
            elif ml_model == 'lr':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/6718043', allow_redirects=True)
                model_path = './pretrained/lr_german.pt'
                open(model_path, 'wb').write(r.content)
                model = LogisticRegression(input_dim=inputs.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        elif data_name == 'heloc':
            if ml_model == 'ann':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/6718045', allow_redirects=True)
                model_path = './pretrained/ann_heloc.pt'
                open(model_path, 'wb').write(r.content)
                model = model_ann.ANN_softmax(input_layer=inputs.shape[1],
                                              hidden_layer_1=100,
                                              num_of_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
            elif ml_model == 'lr':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/6718046', allow_redirects=True)
                model_path = './pretrained/lr_heloc.pt'
                open(model_path, 'wb').write(r.content)
                model = LogisticRegression(input_dim=inputs.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        elif data_name == 'rcdv':
            if ml_model == 'ann':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/7093738', allow_redirects=True)
                model_path = './pretrained/ann_rcdv.pt'
                open(model_path, 'wb').write(r.content)
                model = model_ann.ANN_softmax(input_layer=inputs.shape[1],
                                              hidden_layer_1=100,
                                              num_of_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
            elif ml_model == 'lr':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/7093736', allow_redirects=True)
                model_path = './pretrained/lr_rcdv.pt'
                open(model_path, 'wb').write(r.content)
                model = LogisticRegression(input_dim=inputs.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        elif data_name == 'lending-club':
            if ml_model == 'ann':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/6990764', allow_redirects=True)
                model_path = './pretrained/ann_lending-club.pt'
                open(model_path, 'wb').write(r.content)
                model = model_ann.ANN_softmax(input_layer=inputs.shape[1],
                                              hidden_layer_1=100,
                                              num_of_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
            elif ml_model == 'lr':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/6990766', allow_redirects=True)
                model_path = './pretrained/lr_lending-club.pt'
                open(model_path, 'wb').write(r.content)
                model = LogisticRegression(input_dim=inputs.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        elif data_name == 'student':
            if ml_model == 'ann':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/7093735', allow_redirects=True)
                model_path = './pretrained/ann_student.pt'
                open(model_path, 'wb').write(r.content)
                model = model_ann.ANN_softmax(input_layer=inputs.shape[1],
                                              hidden_layer_1=100,
                                              num_of_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
            elif ml_model == 'lr':
                r = requests.get('https://dataverse.harvard.edu/api/access/datafile/7093732', allow_redirects=True)
                model_path = './pretrained/lr_student.pt'
                open(model_path, 'wb').write(r.content)
                model = LogisticRegression(input_dim=inputs.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            raise NotImplementedError(
                 'The current version of >LoadModel< does not support this data set.')
            
    else:
        raise NotImplementedError(
             'The current version of >LoadModel< does not support training a ML model from scratch, yet.')

    return model
