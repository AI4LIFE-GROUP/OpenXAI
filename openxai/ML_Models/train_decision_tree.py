import ipdb
import time
import numpy as np
import torch, torch.nn as nn
from IPython.display import clear_output
from qhoptim.pyt import QHAdam
import torch.nn.functional as F
from xai_benchmark.ML_Models import NODE
import xai_benchmark.ML_Models.data_loader as loader


dataset_train = loader.DataLoader_Tabular(path='../Data_Sets/COMPAS/', filename='compas-scores-train.csv', label='risk')
dataset_test = loader.DataLoader_Tabular(path='../Data_Sets/COMPAS/', filename='compas-scores-test.csv', label='risk')

input_size = dataset_train.get_number_of_features()

model = nn.Sequential(NODE.DenseBlock(input_size, 2048, num_layers=1, tree_dim=3, depth=6, flatten_output=False, choice_function=NODE.entmax15, bin_function=NODE.entmoid15), NODE.Lambda(lambda x: x[..., 0].mean(dim=-1)), )

optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998) }
exp_name = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format('testing_node', *time.gmtime()[:5])

trainer = NODE.Trainer(
    model=model, loss_function=F.mse_loss,
    experiment_name=exp_name,
    warm_start=False,
    Optimizer=QHAdam,
    optimizer_params=optimizer_params,
    verbose=True,
    n_last_checkpoints=5
)

loss_history, mse_history = [], []
best_mse = float('inf')
best_step_mse = 0
early_stopping_rounds = 5000
report_frequency = 100

# ipdb.set_trace()
# data = lib.Dataset("YEAR", random_state=1337, quantile_transform=True, quantile_noise=1e-3)

for batch in NODE.iterate_minibatches(dataset_train.data.astype('double'), np.array(dataset_train.targets).astype('double'), batch_size=512, shuffle=True, epochs=float('inf')):
    # ipdb.set_trace()
    metrics = trainer.train_on_batch(*batch, device='cpu')
    
    loss_history.append(metrics['loss'])

    if trainer.step % report_frequency == 0:
        trainer.save_checkpoint()
        trainer.average_checkpoints(out_tag='avg')
        trainer.load_checkpoint(tag='avg')
        mse = trainer.evaluate_mse(
            dataset_train.data, dataset_train.targets, device='cpu', batch_size=128)

        if mse < best_mse:
            best_mse = mse
            best_step_mse = trainer.step
            trainer.save_checkpoint(tag='best_mse')
        mse_history.append(mse)
        
        trainer.load_checkpoint()  # last
        trainer.remove_old_temp_checkpoints()

        clear_output(True)
        print("Loss %.5f" % (metrics['loss']))
        print("Val MSE: %0.5f" % (mse))
    if trainer.step > best_step_mse + early_stopping_rounds:
        print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
        print("Best step: ", best_step_mse)
        print("Best Val MSE: %0.5f" % (best_mse))
        break


trainer.load_checkpoint(tag='best_mse')
mse = trainer.evaluate_mse(dataset_test.data, dataset_test.targets, device='cpu')
print('Best step: ', trainer.step)
print("Test MSE: %0.5f" % (mse))
