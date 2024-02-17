import torch
import os
from openxai.train import train_model

# add to config
mean_pred_bounds = {
    'adult': 0.15, 'compas': 0.93, 'gaussian': 0.4, 'german': 0.9,
    'gmsc': 0.96, 'heart': 0.1, 'heloc': 0.4, 'pima': 0.3
}
batch_sizes = {
    'adult': 256, 'compas': 32, 'gaussian': 32, 'german': 16,
    'gmsc': 256, 'heart': 16, 'heloc': 32, 'pima': 16
}
pos_class_weights = {
    'adult': 0.55, 'compas': 0.4, 'gaussian': 0.5, 'german': 0.4,
    'gmsc': 0.25, 'heart': 0.75, 'heloc': 0.5, 'pima': 0.65
}

def main(ml_models=['lr', 'ann'], names = ['adult', 'compas', 'gaussian', 'german', 'gmsc', 'heart', 'heloc', 'pima'],
         epochs = 100, learning_rate = 0.001, scaler = 'minmax', seed = 0, warmup = 5, verbose=False):
    for ml_model in ml_models:
        for name in names:
            print(f'Training {ml_model} on {name} dataset')

            # Train Model
            batch, pcw, mpb = batch_sizes[name], pos_class_weights[name], mean_pred_bounds[name]
            model, best_acc, best_epoch = train_model(ml_model, name, learning_rate, epochs, batch,
                                                      scaler=scaler, seed=seed, pos_class_weight=pcw,
                                                      mean_prediction_bound=mpb, warmup=warmup, verbose=verbose)
            
            # Save Model
            params = {'ep': epochs, 'lr': learning_rate, 'batch': batch, 'seed': seed, 'pcw': pcw,
                      'mpb': mpb, 'wu': warmup, 'acc': str(round(best_acc*100, 2)), 'at_ep': best_epoch}
            params_str = '_'.join([f'{k}_{v}' for k, v in params.items()])
            model_file_name = f'{name}_{ml_model}_{scaler}_{params_str}.pt'
            model_folder_name = f'models/{model.name}/'
            if not os.path.exists(model_folder_name):
                os.makedirs(model_folder_name)
            torch.save(model.state_dict(),  model_folder_name + model_file_name)
            print(f'File saved to {model_folder_name + model_file_name}')

if __name__ == "__main__":
    main()