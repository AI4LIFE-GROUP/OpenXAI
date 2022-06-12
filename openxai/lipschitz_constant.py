import torch
import numpy as np

# models
import ML_Models.ANN.model as model_ann
from xai_benchmark.ML_Models.SVM.model import SVM, SVM_Loss
from xai_benchmark.ML_Models.LR.model import LogisticRegression

# (Train & Test) Loaders
import ML_Models.data_loader as loaders


def main(data_names: list = ['adult', 'compas', 'german'],
         model_name: str = 'lr', data_loader_batch_size: int = 30):

    '''
    LOAD ML MODELS
    '''
    
    maxweight_dict = {
        'adult': 0,
        'german': 0,
        'compas': 0
    }
    
    for data_name in data_names:
        
        loader_train, loader_test = loaders.return_loaders(data_name=data_name, is_tabular=True,
                                                           batch_size=data_loader_batch_size)
        data_iter = iter(loader_test)
        inputs, labels = data_iter.next()
    
        if data_name == 'adult':
            if model_name == 'ann':
                model_path = './ML_Models/Saved_Models/ANN/adult_lr_0.002_acc_0.83.pt'
                ann = model_ann.ANN_softmax(input_layer=inputs.shape[1], hidden_layer_1=100, num_of_classes=2)
                ann.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
                model = ann
                L_map = ann.L_relu
        
            elif model_name == 'lr':
                model_path = './ML_Models/Saved_Models/LR/adult_lr_0.002_acc_0.84.pt'
                lr = LogisticRegression(input_dim=inputs.shape[1])
                lr.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
                model = lr
                L_map = lr.linear
    
        elif data_name == 'compas':
            if model_name == 'ann':
                model_path = './ML_Models/Saved_Models/ANN/compas_lr_0.002_acc_0.85.pt'
                ann = model_ann.ANN_softmax(input_layer=inputs.shape[1], hidden_layer_1=100, num_of_classes=2)
                ann.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
                model = ann
                L_map = ann.L_relu
        
            elif model_name == 'lr':
                model_path = './ML_Models/Saved_Models/LR/compas_lr_0.002_acc_0.85.pt'
                lr = LogisticRegression(input_dim=inputs.shape[1])
                lr.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
                model = lr
                L_map = lr.linear
    
        elif data_name == 'german':
            if model_name == 'ann':
                model_path = './ML_Models/Saved_Models/ANN/german_lr_0.002_acc_0.71.pt'
                ann = model_ann.ANN_softmax(input_layer=inputs.shape[1], hidden_layer_1=100, num_of_classes=2)
                ann.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
                model = ann
                L_map = ann.L_relu
        
            elif model_name == 'svm':
                dataset_train = loaders.DataLoader_Tabular(
                    path='./Data_Sets/German_Credit_Data/',
                    filename='german-train.csv', label='credit-risk')
            
                model_path = './ML_Models/Saved_Models/SVM/german_svm_0.002_acc_0.73.pt'
                svm = SVM(dataset_train.get_number_of_features(), num_of_classes=2)
                svm.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
                model = svm
                L_map = svm.input1
        
            elif model_name == 'lr':
                model_path = './ML_Models/Saved_Models/LR/german_lr_0.002_acc_0.72.pt'
                lr = LogisticRegression(input_dim=inputs.shape[1])
                lr.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model = lr
                L_map = lr.linear
        
        # get model weights &
        # compute maximum model weight
        if model_name == 'ann':
            weights_hidden = model._modules['input1'].weight.detach().numpy()
            weights_last = model._modules['input2'].weight.detach().numpy()
            max1 = np.max(weights_hidden)
            max2 = np.max(weights_last)
            max_weight = np.max([max1, max2]).round(4)
            
        elif model_name == 'lr':
            weights_last = model._modules['linear'].weight.detach().numpy()
            max_weight = np.max(weights_last).round(4)

        maxweight_dict[data_name] = max_weight
    
    print('Maximum model weights on ' + model_name + ' classifiers:')
    print(maxweight_dict)
    
    
if __name__ == "__main__":
    main()
