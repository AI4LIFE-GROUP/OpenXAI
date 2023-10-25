# Utils
import torch
import numpy as np
import pickle
from sklearn.metrics import auc

# ML models
from openxai.LoadModel import LoadModel

# Data loaders
from openxai.dataloader import return_loaders

# Explanation models
from openxai.Explainer import Explainer

# Evaluation methods
from openxai.evaluator import Evaluator

# Perturbation methods required for the computation of the relative stability metrics
from openxai.explainers.catalog.perturbation_methods import NormalPerturbation
from openxai.explainers.catalog.perturbation_methods import NewDiscrete_NormalPerturbation

# Choose the model and the data set you wish to generate explanations for
data_loader_batch_size = 10
data_name = 'heloc'  # must be one of ['heloc', 'adult', 'german', 'compas']
model_name = 'lr'  # must be one of ['lr', 'ann']

"""### (0) Explanation method hyperparameters"""

# Hyperparameters for Lime
lime_mode = 'tabular'
lime_sample_around_instance = True
lime_kernel_width = 0.75
lime_n_samples = 1000
lime_discretize_continuous = False
lime_standard_deviation = float(np.sqrt(0.03))

"""### (1) Data Loaders"""

# Get training and test loaders
loader_train, loader_test = return_loaders(data_name=data_name,
                                           download=True,
                                           batch_size=data_loader_batch_size)
data_iter = iter(loader_test)
inputs, labels = data_iter.next()
labels = labels.type(torch.int64)

# get full training data set
data_all = torch.FloatTensor(loader_train.dataset.data)

"""### (2) Load a pretrained ML model"""

# Load pretrained ml model
model = LoadModel(data_name=data_name,
                  ml_model=model_name,
                  pretrained=True)

"""### (3) Choose an explanation method

# """#### Explanation method with default hyperparameters"""

# You can also use the default hyperparameters like so:

control = Explainer(method='control',
           model=model,
           dataset_tensor=data_all)
control_default_exp = control.get_explanation(inputs.float(),
                                label=labels)

grad = Explainer(method='grad',
           model=model,
           dataset_tensor=data_all,
           param_dict_grad=None)
grad_default_exp = grad.get_explanation(inputs.float(),
                                label=labels)

ig = Explainer(method='ig',
           model=model,
           dataset_tensor=data_all,
           param_dict_ig=None)
ig_default_exp = ig.get_explanation(inputs.float(),
                                label=labels)

itg = Explainer(method='itg',
           model=model,
           dataset_tensor=data_all)
itg_default_exp = itg.get_explanation(inputs.float(),
                                label=labels)

sg = Explainer(method='sg',
           model=model,
           dataset_tensor=data_all,
           param_dict_sg=None)
sg_default_exp = sg.get_explanation(inputs.float(),
                                label=labels)

lime = Explainer(method='lime',
                 model=model,
                 dataset_tensor=data_all,
                 param_dict_lime=None)
lime_default_exp = lime.get_explanation(inputs.float(),
                                        label=labels)

shap = Explainer(method='shap',
           model=model,
           dataset_tensor=data_all,
           param_dict_shap=None)
shap_default_exp = shap.get_explanation(inputs.float(),
                                label=labels)

explainers = [control, grad, ig, itg, sg, shap, lime]
explanations = [control_default_exp, grad_default_exp, ig_default_exp, itg_default_exp, sg_default_exp, shap_default_exp, lime_default_exp]
algos = ['control', 'grad', 'ig', 'itg', 'sg', 'shap', 'lime']

def generate_mask(explanation, top_k):
    if not isinstance(explanation, torch.Tensor):
        explanation = torch.Tensor(explanation)
    mask_indices = torch.topk(explanation.abs(), top_k).indices
    mask = torch.ones(explanation.shape, dtype=bool)
    for i in mask_indices:
        mask[i] = False
    return mask


# Perturbation class parameters
perturbation_mean = 0.0
perturbation_std = 0.05
perturbation_flip_percentage = 0.03
if data_name == 'compas':
    feature_types = ['c', 'd', 'c', 'c', 'd', 'd', 'd']
# Adult feature types
elif data_name == 'adult':
    feature_types = ['c'] * 6 + ['d'] * 7
# Gaussian feature types
elif data_name == 'synthetic':
    feature_types = ['c'] * 20
# Heloc feature types
elif data_name == 'heloc':
    feature_types = ['c'] * 23
# German Credit Data feature metadata
elif data_name == 'german':
    feature_types = ['c'] * 8 + ['d'] * 12
    feature_metadata = dict()
    feature_metadata['feature_n_cols'] = [1, 1, 1, 1, 1, 1, 1, 1, 4, 5, 10, 5, 5, 4, 3, 4, 3, 3, 4, 2]
    feature_metadata['feature_types'] = feature_types
    feature_types = feature_metadata

# Perturbation methods
if data_name == 'german':
    # use special perturbation class
    perturbation = NewDiscrete_NormalPerturbation("tabular",
                                                  mean=perturbation_mean,
                                                  std_dev=perturbation_std,
                                                  flip_percentage=perturbation_flip_percentage)

else:
    perturbation = NormalPerturbation("tabular",
                                      mean=perturbation_mean,
                                      std_dev=perturbation_std,
                                      flip_percentage=perturbation_flip_percentage)


"""### (4) Choose an evaluation metric"""

for explainer, explanation_x, algo in zip(explainers, explanations, algos):
    # PRA_AUC = []
    # RC_AUC = []
    # FA_AUC = []
    # RA_AUC = []
    # SA_AUC = []
    # SRA_AUC = []
    PGU_AUC = []
    PGI_AUC = []
    for index in range(data_loader_batch_size):
        print('iteration:', index)

        input_dict = dict()

        # inputs and models
        input_dict['x'] = inputs[index].reshape(-1)
        # print(input_dict['x'])
        input_dict['input_data'] = inputs
        input_dict['explainer'] = explainer
        # print(explainer)
        input_dict['explanation_x'] = explanation_x[index, :].flatten()
        # print(input_dict['explanation_x'])
        input_dict['model'] = model

        # perturbation method used for the stability metric
        input_dict['perturbation'] = perturbation
        input_dict['perturb_method'] = perturbation
        input_dict['perturb_max_distance'] = 0.4
        input_dict['feature_metadata'] = feature_types
        input_dict['p_norm'] = 2
        input_dict['eval_metric'] = None

        # gt label and model prediction
        input_dict['y'] = labels[index].detach().item()
        input_dict['y_pred'] = torch.max(model(inputs[index].unsqueeze(0).float()), 1).indices.detach().item()

        # required for the representation stability measure
        input_dict['L_map'] = model

        # PRA = []
        # RC = []
        # FA = []
        # RA = []
        # SA = []
        # SRA = []
        PGU = []
        PGI = []
        # RIS = []
        # ROS = []
        # RRS = []

        auc_x = np.arange(1, input_dict['explanation_x'].shape[0]+1) / input_dict['explanation_x'].shape[0]

        for topk in range(1, input_dict['explanation_x'].shape[0]+1):

            # topk and mask
            input_dict['top_k'] = topk
            input_dict['mask'] = generate_mask(input_dict['explanation_x'].reshape(-1), input_dict['top_k'])

            evaluator = Evaluator(input_dict,
                                  inputs=inputs,
                                  labels=labels,
                                  model=model,
                                  explainer=lime)

            # if hasattr(model, 'return_ground_truth_importance'):
            #     # evaluate prediction gap on important features
            #     PRA.append(evaluator.evaluate(metric='PRA')[1])
            #     # print('PRA:', type(PRA[-1]))

            #     # evaluate rank correlation
            #     RC.append(evaluator.evaluate(metric='RC')[1])
            #     # print('RC:', RC[-1])

            #     # evaluate feature agreement
            #     FA.append(evaluator.evaluate(metric='FA')[1])
            #     # print('FA:', FA[-1])

            #     # evaluate rank agreement
            #     RA.append(evaluator.evaluate(metric='RA')[1])
            #     # print('RA:', RA[-1])

            #     # evaluate sign agreement
            #     SA.append(evaluator.evaluate(metric='SA')[1])
            #     # print('SA:', SA[-1])

            #     # evaluate signed rankcorrelation
            #     SRA.append(evaluator.evaluate(metric='SRA')[1])
                # print('SRA:', SRA[-1])

            # evaluate prediction gap on umportant features
            PGU.append(evaluator.evaluate(metric='PGU'))
            # print('PGU:', PGU[-1])

            # evaluate prediction gap on important features
            PGI.append(evaluator.evaluate(metric='PGI'))
            # print('PGI:', PGI[-1])

        # # evaluate prediction gap on important features
        # RIS.append(evaluator.evaluate(metric='RIS'))
        # print('RIS:', RIS[-1])

        # # evaluate prediction gap on important features
        # ROS.append(evaluator.evaluate(metric='ROS'))
        # print('ROS:', ROS[-1])

        # # evaluate prediction gap on important features
        # RRS.append(evaluator.evaluate(metric='RRS'))
        # print('RRS:', RRS[-1])

        # PRA_AUC.append(auc(auc_x, PRA))
        # RC_AUC.append(auc(auc_x, RC))
        # FA_AUC.append(auc(auc_x, FA))
        # RA_AUC.append(auc(auc_x, RA))
        # SA_AUC.append(auc(auc_x, SA))
        # SRA_AUC.append(auc(auc_x, SRA))
        PGU_AUC.append(auc(auc_x, PGU))
        PGI_AUC.append(auc(auc_x, PGI))

    print('--- MEAN ---')
    # print('PRA', np.mean(PRA_AUC))
    # print('RC', np.mean(RC_AUC))
    # print('FA', np.mean(FA_AUC))
    # print('RA', np.mean(RA_AUC))
    # print('SA', np.mean(SA_AUC))
    # print('SRA', np.mean(SRA_AUC))
    print('PGU', np.mean(PGU_AUC))
    print('PGI', np.mean(PGI_AUC))
    print('--- STD ---')
    # print('PRA', np.std(PRA_AUC))
    # print('RC', np.std(RC_AUC))
    # print('FA', np.std(FA_AUC))
    # print('RA', np.std(RA_AUC))
    # print('SA', np.std(SA_AUC))
    # print('SRA', np.std(SRA_AUC))
    print('PGU', np.std(PGU_AUC))
    print('PGI', np.std(PGI_AUC))

    # if hasattr(model, 'return_ground_truth_importance'):
    #     np.save(data_name + '_' + model_name + '_' + algo + '_gtfaithfulness.npy', np.array([]), allow_pickle=False)
    np.save(data_name + '_' + model_name + '_' + algo + '_faithfulness.npy', np.array([[np.mean(PGU_AUC), np.mean(PGI_AUC)], [np.std(PGU_AUC), np.std(PGI_AUC)]]), allow_pickle=False)
