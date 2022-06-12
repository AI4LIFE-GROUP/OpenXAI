import torch
import torch.distributions as tdist
from explainers.perturbation_methods import RandomPerturbation
from explainers.perturbation_methods import UniformPerturbation
from explainers.perturbation_methods import BootstrapPerturbation
from explainers.perturbation_methods import MarginalPerturbation
from explainers.perturbation_methods import AdversarialPerturbation
from explainers.perturbation_methods import NormalPerturbation
from torch import nn

def compute_distances(original_point, perturbations, max_distance) :
    pdist = nn.PairwiseDistance(p=2)
    distances = pdist(original_point, perturbations)
    return sum([i > max_distance for i in distances]) == 0


def test_perturbation_methods():
    print("Generating Normal Perturbations....")
    perturbation_method = NormalPerturbation("tabular")
    original_sample = torch.tensor([1, 0.3, 0, 0.41, 0.51], dtype=torch.float32)
    feature_mask = torch.BoolTensor([0, 1, 0, 1, 0])
    feature_type = ['d', 'c', 'd', 'c', 'c']
    data_samples = torch.tensor([[1, 2, 3, 5, 1], [-1, -2, 4, 2, 1], [2, -2, 1, 3, 4]])
    num_samples = 10
    max_distance = 100
    perturbations = perturbation_method.get_perturbed_inputs(original_sample, feature_mask, num_samples, feature_type=feature_type)
    print("Success! Dimensions match") if (len(perturbations)) <= num_samples else print("Dimensions Mismatch")
    print(perturbations)



    print("Generating Random Perturbations....")
    perturbation_method = RandomPerturbation("tabular")
    original_sample = torch.tensor([1,21,3,41,4.5], dtype = torch.float32)
    feature_mask = torch.BoolTensor([0,1,0,1,0])
    num_samples = 10
    max_distance = 4
    perturbations = perturbation_method.get_perturbed_inputs( original_sample, feature_mask, num_samples, max_distance)
    print("Success! ") if (compute_distances(original_sample, perturbations, max_distance) == True) else print("Test Failed! ")

    print("Generating Uniform Perturbations....")
    perturbation_method = UniformPerturbation("tabular")
    original_sample = torch.tensor([1, 21, 3, 41, 4.5], dtype=torch.float32)
    feature_mask = torch.BoolTensor([0, 1, 0, 1, 0])
    num_samples = 10
    max_distance = 4
    perturbations = perturbation_method.get_perturbed_inputs(original_sample, feature_mask, num_samples, max_distance)
    print("Success! ") if (compute_distances(original_sample, perturbations, max_distance) == True) else print("Test Failed! ")

    # Testing marginal distribution based perturbation method
    print("Generating Marginal Perturbations....")
    perturbation_method = MarginalPerturbation("tabular", [tdist.Normal(0,5), tdist.Normal(1,6)])
    original_sample = torch.tensor([1, 21], dtype=torch.float32)
    feature_mask = torch.BoolTensor([0, 1])
    num_samples = 10
    max_distance = 3
    # print(perturbation_method.get_perturbed_inputs(original_sample, feature_mask, num_samples, max_distance))
    perturbations = perturbation_method.get_perturbed_inputs(original_sample, feature_mask, num_samples, max_distance)
    print("Success! ") if (compute_distances(original_sample, perturbations, max_distance) == True) else print("Test Failed! ")

    print("Generating Bootstrap Perturbations....")
    perturbation_method = BootstrapPerturbation("tabular")
    original_sample = torch.tensor([1, 21, 3, 41, 4.5], dtype=torch.float32)
    feature_mask = torch.BoolTensor([0, 1, 0, 1, 0])
    data_samples = torch.tensor([[1,2,3,5,1], [-1,-2,4,2,1],[2,-2,1,3,4]])
    num_samples = 10
    max_distance = 100
    perturbations = perturbation_method.get_perturbed_inputs(original_sample, feature_mask, num_samples, max_distance, data_samples)
    print("Success! Dimensions match") if (len(perturbations)) <= num_samples else print("Dimensions Mismatch")
    print("Success! ") if (compute_distances(original_sample, perturbations, max_distance) == True) else print("Test Failed! ")

    pass


if __name__ == '__main__':
    test_perturbation_methods()
