
# ANN models
import torch
import ML_Models.ANN.model as model_ann

# (Train & Test) Loaders
import ML_Models.data_loader as loaders

# Explanation Models
import explainers.lime as lime

import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F

def main():
    
    '''
    Loading Data Loaders
    '''

    params = {
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
    
    loader_train_cifar, loader_test_mnist = loaders.return_loaders(data_name='cifar10', is_tabular=False, batch_size=1)
    loader_train_adult, loader_test_adult = loaders.return_loaders(data_name='adult', is_tabular=True, batch_size=5)
    loader_train_gauss, loader_test_gauss = loaders.return_loaders(data_name='gaussian', is_tabular=True,
                                                                   batch_size=5, gauss_params=params)

    
    '''
    Gaussian Data DGP
    '''
    
    gauss_train_input = loader_train_gauss.dataset.ground_truth_dict
    data_iter = iter(loader_train_gauss)
    input, labels, weights, masks, masked_weights, probs, cluster_idx = data_iter.next()

    model_path = 'ML_Models/Saved_Models/ANN/gaussian_lr_0.002_acc_0.89.pt'
    ann = model_ann.ANN(20, hidden_layer_1=100, num_of_classes=1)
    ann.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    a = ann(input.float())
    b = ann.predict(input.float())


    # Test Out Tabular Data
    data_iter = iter(loader_train_gauss)
    input, labels, weights, masks, masked_weights, probs, cluster_idx = data_iter.next()
    input = input.numpy()
    labels = labels.numpy()
    masked_weights = masked_weights.numpy()

    '''
    Testing T-LIME on Gaussian Data
    '''
    # full training data set
    gaussian_all = loader_train_gauss.dataset.data
    
    # requires the full training data as an input
    # requires ann.predict or ann.predict_proba to be numpy compatible
    init_lime = lime.LIME(ann.predict, gaussian_all, mode="tabular", kernel_width=0.75, n_samples=1000)

    # also takes batches
    explantions = init_lime.explain_batch(input)
    
 
    '''
    currently, we are returning numpy arrays for tabular data
    '''
    # tabular data: in case your input needs data or labels
    adult_train_input = loader_train_adult.dataset.data
    adult_train_output = loader_train_adult.dataset.targets
    
    # in case you need the feature names or target name
    feat_names = loader_train_adult.dataset.feature_names
    target_name = loader_train_adult.dataset.target_name

    # image data: in case your input needs data or labels
    '''
    currently, we are returning pytorch tensors for image data
    '''
    mnist_train_input = loader_train_cifar.dataset.data
    mnist_train_output = loader_train_cifar.dataset.targets
    
    '''
    Testing ANNs, CNNs and Data Loaders
    '''
    
    model_path = 'ML_Models/Saved_Models/ANN/cifar10_lr_0.02_acc_0.20.pt'
    cnn = model_ann.CNN(n_channels=3, image_size=32, kernel_size=5)
    cnn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Test Out Image Data
    data_iter = iter(loader_train_cifar)
    input_image, labels = data_iter.next()
    labels = labels.numpy()

    input = torch.squeeze(input_image)
    im = transforms.ToPILImage()(input).convert("RGB")
    
    test_output11 = cnn(torch.tensor(input_image))
    test_output12 = cnn.predict_proba(input_image)

    model_path = 'ML_Models/Saved_Models/ANN/adult_lr_0.002_acc_0.84.pt'
    ann = model_ann.ANN(20, hidden_layer_1=5, num_of_classes=1)
    ann.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Test Out Tabular Data
    data_iter = iter(loader_train_adult)
    input, labels = data_iter.next()
    input = input.numpy()
    labels = labels.numpy()
    
    test_output21 = ann(torch.tensor(input).float())
    test_output22 = ann.predict_proba(input)
    
    '''
    Testing T-LIME
    '''
    # full training data set
    adult_all = loader_train_adult.dataset.data
    
    # requires the full training data as an input
    # requires ann.predict or ann.predict_proba to be numpy compatible
    init_lime = lime.LIME(ann.predict, adult_all, mode="tabular", kernel_width=0.75, n_samples=1000)
    
    # also takes batches
    explantions = init_lime.explain_batch(input)

    '''
    Testing I-LIME
    '''
    # full training data set
    cifar_all = loader_train_cifar.dataset.data
    cifar_targets = loader_train_cifar.dataset.targets

    def get_image(path):
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    img = get_image('./explainers/dogs.png')

    def get_pil_transform():
        transf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224)
        ])
    
        return transf

    def get_preprocess_transform():
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transf = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
        return transf

    def get_preprocess_transform2():

        transf = transforms.Compose([
            transforms.ToTensor()
        ])
    
        return transf

    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()
    preprocess_transform2 = get_preprocess_transform2()


    a = [pill_transf(img)]
    
    b = [im]
    

    # requires the full training data as an input
    # requires ann.predict or ann.predict_proba to be numpy compatible
    # init_lime = lime.LIME(cnn.predict_proba, cifar_all, mode="images")

    def batch_predict(images):
        cnn.eval()
        batch = torch.stack(tuple(preprocess_transform2(i) for i in images), dim=0)
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn.to(device)
        batch = batch.to(device)
    
        probs = cnn(batch)
        return probs.detach().cpu().numpy()

    test_pred = batch_predict(b)

    init_lime = lime.LIME(batch_predict, cifar_all, mode='image')
    # also takes batches
    t = np.array(im)
    explantions = init_lime.explain_batch(np.array(im))

if __name__ == "__main__":
    # execute only if run as a script
    main()