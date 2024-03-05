import torch
import os
from openxai.model import train_model
from openxai.experiment_utils import load_config

if __name__ == "__main__":
    config = load_config('experiment_config.json')
    model_names, data_names = config['model_names'], config['data_names']
    train_config = config['training']
    epochs, learning_rate = train_config['epochs'], train_config['learning_rate']
    scaler, seed, warmup = train_config['scaler'], train_config['seed'], train_config['warmup']
    for model_name in model_names:
        for data_name in data_names:
            print(f'Training {model_name} on {data_name} dataset')

            # Train Model
            data_config = train_config[data_name]
            batch, pcw, mpb = data_config['batch_size'], data_config['pos_class_weight'], data_config['mean_pred_bound']
            model, best_acc, best_epoch = train_model(model_name, data_name, learning_rate, epochs, batch,
                                                      scaler=scaler, seed=seed, pos_class_weight=pcw,
                                                      mean_prediction_bound=mpb, warmup=warmup, verbose=False)
            
            # Save Model
            params = {'ep': epochs, 'lr': learning_rate, 'batch': batch, 'seed': seed, 'pcw': pcw,
                      'mpb': mpb, 'wu': warmup, 'acc': str(round(best_acc*100, 2)), 'at_ep': best_epoch}
            params_str = '_'.join([f'{k}_{v}' for k, v in params.items()])
            model_file_name = f'{data_name}_{model_name}_{scaler}_{params_str}.pt'
            model_folder_name = f'models/{model.name}/'
            if not os.path.exists(model_folder_name):
                os.makedirs(model_folder_name)
            torch.save(model.state_dict(),  model_folder_name + model_file_name)
            print(f'File saved to {model_folder_name + model_file_name}')