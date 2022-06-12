import numpy as np

## Data loader parameters
data_loader_batch_size = 100

## Percentage most important features
percentage_most_important = 0.25

## SHAP parameters
shap_subset_size = 500

## LIME parameters
lime_mode = 'tabular'
lime_sample_around_instance = True
lime_kernel_width = 0.75
lime_n_samples = 1000
lime_discretize_continuous = False
lime_standard_deviation_003 = float(np.sqrt(0.03))
lime_standard_deviation_005 = float(np.sqrt(0.05))
lime_standard_deviation_01 = float(np.sqrt(0.1))

## Gradient parameters
grad_absolute_value = False

## SmoothGrad parameters
sg_n_samples = 500
sg_standard_deviation_003 = float(np.sqrt(0.03))
sg_standard_deviation_005 = float(np.sqrt(0.05))
sg_standard_deviation_01 = float(np.sqrt(0.1))

## Integraded Gradient parameters
ig_method = 'gausslegendre'
ig_multiply_by_inputs = False
ig_baseline = None  # This parameter will be inferred later on

## Perturbation class parameters
perturbation_mean = 0.0
perturbation_std = 0.05
perturbation_flip_percentage = 0.03
perturbation_max_distance = 0.4