import numpy as np
import torch
import torch.distributions as tdist
from typing import Any, Callable, Tuple, Union, cast
from torch import Tensor
from torch import nn
from random import random

class BasePerturbation:
    '''
    Base Class for perturbation methods.
    '''

    def __init__(self, data_format):
        '''
        Initialize generic parameters for the perturbation method
        '''
        self.data_format = data_format


    def get_perturbed_inputs(self):
        '''
        This function implements the logic of the perturbation methods which will return perturbed samples.
        '''
        pass





class UniformPerturbation(BasePerturbation):
    def __init__(self, data_format):
        super(UniformPerturbation, self).__init__(data_format)


    def get_perturbed_inputs(self, original_sample : torch.FloatTensor, feature_mask : torch.BoolTensor, num_samples : int, max_distance : int) -> torch.tensor:
        '''

        feature mask : this indicates the static features
        num_samples : number of perturbed samples.
        max_distance : the maximum distance between original sample and purturbed samples.
        Algorithmic Resource Link : http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        '''
        assert len(feature_mask) == len(
            original_sample), "mask size == original sample in get_perturbed_inputs for {}".format(self.__class__)
        gaussian_generator = tdist.Normal(torch.tensor(torch.empty(original_sample.shape).fill_(0)), torch.tensor(torch.empty(original_sample.shape).fill_(1)))
        gaussian_samples = gaussian_generator.sample((num_samples,))
        gaussian_samples_normed = gaussian_samples / torch.norm(gaussian_samples, dim = 1)[:, None]
        radius_sample = torch.pow(torch.rand(num_samples) * max_distance, 1/(len(original_sample.flatten()) - 1))
        gaussian_samples_normed = gaussian_samples_normed * radius_sample[:, None]
        return original_sample + gaussian_samples_normed * (~feature_mask)


        
class RandomPerturbation(BasePerturbation):
    def __init__(self, data_format, dist = 'gaussian'):
        super(RandomPerturbation, self).__init__(data_format)

        self.dist = dist

    def get_perturbed_inputs(self, original_sample : torch.FloatTensor, feature_mask : torch.BoolTensor , num_samples : int, max_distance : int) -> torch.tensor:
        '''

        feature mask : this indicates the static features
        num_samples : number of perturbed samples.
        max_distance : the maximum distance between original sample and purturbed samples.
        '''
        assert len(feature_mask) == len(original_sample) , "mask size == original sample in get_perturbed_inputs for {}".format(self.__class__)

        if self.dist == 'gaussian':
            gaussian_generator = tdist.Normal(torch.tensor(torch.empty(original_sample.shape).fill_(0)),
                                              torch.tensor(torch.empty(original_sample.shape).fill_(1)))
            gaussian_samples = gaussian_generator.sample((num_samples,))
            gaussian_samples_normed = gaussian_samples / torch.norm(gaussian_samples, dim=1)[:, None]
            radius_sample = torch.rand(num_samples) * max_distance
            gaussian_samples_normed = gaussian_samples_normed * radius_sample[:, None]
            return original_sample + gaussian_samples_normed * (~feature_mask)




class BootstrapPerturbation(BasePerturbation):
    def __init__(self, data_format):
        super(BootstrapPerturbation, self).__init__(data_format)

    def _filter_out_of_range_samples(self, original_sample, perturbed_samples, max_distance):
        '''
        This method is used to filter out samples that are not within \epsilon norm ball.
        perturbed_samples : unfiltered samples
        '''
        original_sample_unsqueezed = original_sample.unsqueeze(0)
        distance_from_original_sample_unsqueezed = torch.cdist(perturbed_samples, original_sample_unsqueezed, p=2.0)
        perturbed_samples = perturbed_samples* (distance_from_original_sample_unsqueezed <= max_distance)
        return perturbed_samples[perturbed_samples.sum(dim = 1) != 0]


    def _get_samples_within_norm_ball(self, original_sample, data_samples, max_distance):
        '''
        Returns samples from data_samples that are within max_distance L2 distance from original sample.
        '''
        pdist = nn.PairwiseDistance(p=2)
        distances = pdist(original_sample, data_samples)
        return [i <= max_distance for i in distances]



    def get_perturbed_inputs(self, original_sample : torch.FloatTensor, feature_mask : torch.BoolTensor, num_samples : int, max_distance : int, data_samples : torch.tensor) -> torch.tensor:
        '''

        feature mask : this indicates the static features
        num_samples : number of perturbed samples.
        max_distance : the maximum distance between original sample and purturbed samples.


        '''
        perturbed_samples = []

        # Extract samples from data_samples that within \epsilon norm ball
        samples_within_norm_ball = self._get_samples_within_norm_ball(original_sample, data_samples, max_distance)
        extracted_data_samples = data_samples[torch.BoolTensor(samples_within_norm_ball), :]

        # Randomly pick num_samples number of samples from samples_within_norm_ball to swap values
        perturbed_samples = original_sample * feature_mask + extracted_data_samples * (~feature_mask)
        sampled_perturbations = torch.randperm(len(perturbed_samples))[:min(num_samples, len(data_samples))]
        return perturbed_samples[sampled_perturbations]


class NormalPerturbation(BasePerturbation):
    def __init__(self, data_format, mean: int = 0, std_dev: float = 0.05, flip_percentage: float = 0.3):
        self.mean = mean
        self.std_dev = std_dev
        self.flip_percentage = flip_percentage

        super(NormalPerturbation, self).__init__(data_format)
        '''
        Initializes the marginal perturbation method where each column is sampled from marginal distributions given per variable.
        dist_per_feature : vector of distribution generators (tdist under torch.distributions).
        Note : These distributions are assumed to have zero mean since they get added to the original sample.
        '''
        pass

    def get_perturbed_inputs(self, original_sample: torch.FloatTensor, feature_mask: torch.BoolTensor,
                             num_samples: int, feature_metadata: list, max_distance: int = None) -> torch.tensor:
        '''
        feature mask : this indicates the static features
        num_samples : number of perturbed samples.
        max_distance : the maximum distance between original sample and purturbed samples.
        '''
        feature_type = feature_metadata
        assert len(feature_mask) == len(
            original_sample), "mask size == original sample in get_perturbed_inputs for {}".format(self.__class__)

        perturbed_cols = []
        continuous_features = torch.tensor([i == 'c' for i in feature_type])
        discrete_features = torch.tensor([i == 'd' for i in feature_type])

        # Processing continuous columns
        mean = self.mean
        std_dev = self.std_dev
        perturbations = torch.normal(mean, std_dev,
                                     [num_samples, len(feature_type)]) * continuous_features + original_sample

        # Processing discrete columns
        flip_percentage = self.flip_percentage
        p = torch.empty(num_samples, len(feature_type)).fill_(flip_percentage)
        perturbations = perturbations * (~discrete_features) + torch.abs(
            (perturbations * discrete_features) - (torch.bernoulli(p) * discrete_features))

        # keeping features static where the feature mask is high
        perturbed_samples = original_sample * feature_mask + perturbations * (~feature_mask)

        return perturbed_samples

class NewDiscrete_NormalPerturbation(BasePerturbation):
    def __init__(self, data_format, mean: int = 0, std_dev: float = 0.05, flip_percentage: float = 0.3):
        self.mean = mean
        self.std_dev = std_dev
        self.flip_percentage = flip_percentage

        super(NewDiscrete_NormalPerturbation, self).__init__(data_format)
        '''
        Initializes the marginal perturbation method where each column is sampled from marginal distributions given per variable.
        dist_per_feature : vector of distribution generators (tdist under torch.distributions).
        Note : These distributions are assumed to have zero mean since they get added to the original sample.
        '''
        pass

    def get_perturbed_inputs(self, original_sample: torch.FloatTensor, feature_mask: torch.BoolTensor,
                             num_samples: int, feature_metadata: dict,
                             max_distance: int = None) -> torch.tensor:
        '''
        feature mask : this indicates the static features
        num_samples : number of perturbed samples.
        max_distance : the maximum distance between original sample and purturbed samples.
        feature_type : list containing metadata on which features are continuous vs. discrete
        feature_num_cols: list containing metadata on how many (potentially one-hot encoded)
            columns correspond to each feature
        '''
        
        feature_type = feature_metadata['feature_types']
        feature_num_cols = feature_metadata['feature_n_cols']
        
        assert len(feature_mask) == len(
            original_sample), "mask size == original sample in get_perturbed_inputs for {}".format(self.__class__)
        
        perturbations = original_sample
        original_sample_dim = original_sample.shape[0]
        
        # iterate through features
        feature_ind = 0
        
        for d in range(len(feature_type)):
            first_col_ind = feature_ind
            last_col_ind = feature_ind + feature_num_cols[d]
            feature_ind_mask = torch.tensor([j == feature_ind for j in range(original_sample_dim)])
    
            if feature_type[d] == 'c':
                # continuous feature: add Gaussian noise to original sample
                perturbations = torch.normal(self.mean, self.std_dev,
                             [num_samples, original_sample_dim]) * feature_ind_mask + perturbations
                
            elif feature_type[d] == 'd':
                # discrete feature: flip each sample w.p. p
                
                new_feature_options = []
                # return which column is 1
                for j in range(first_col_ind, last_col_ind):
                    if original_sample[j] != 1:
                        new_feature_options.append(j)
                    else:
                        discrete_col_ind = j

                assert len(new_feature_options) == feature_num_cols[d] - 1
                            
                feature_ind_mask = torch.tensor([j == discrete_col_ind for j in range(original_sample_dim)])
                
                p = torch.empty(num_samples, original_sample_dim).fill_(self.flip_percentage)
                samples_to_flip = torch.bernoulli(p) * feature_ind_mask
                
                # if the feature has 1 value, flip that column w.p. p
                if feature_num_cols[d] > 1:
                    # select 1 of the other columns uniformly at random as the new value w.p. p
                    flip_to_this_col_ind = np.random.choice(new_feature_options, 
                                                          size = num_samples)
                    
                    # choose new column value for samples that are flipped
                    for sample_ind in range(num_samples):
                        samples_to_flip[sample_ind, flip_to_this_col_ind[sample_ind]] = samples_to_flip[sample_ind, discrete_col_ind]
                        
                        assert torch.sum(samples_to_flip[sample_ind, :]) == 0 or torch.sum(samples_to_flip[sample_ind, :]) == 2
                        
                feature_group_mask = torch.tensor([first_col_ind <= j < last_col_ind  for j in range(original_sample_dim)])
                    
                perturbations = perturbations * (~feature_group_mask) + feature_group_mask * torch.abs(perturbations - (samples_to_flip))
                    
            feature_ind += feature_num_cols[d]

        # keeping features static that are in top-K based on feature mask
        perturbed_samples = original_sample * feature_mask + perturbations * (~feature_mask)

        return perturbed_samples

class MarginalPerturbation(BasePerturbation):
    def __init__(self, data_format, dist_per_feature):
        super(MarginalPerturbation, self).__init__(data_format)
        '''
        Initializes the marginal perturbation method where each column is sampled from marginal distributions given per variable.  
        dist_per_feature : vector of distribution generators (tdist under torch.distributions). 
        Note : These distributions are assumed to have zero mean since they get added to the original sample.  
        '''
        assert sum([hasattr(i.__class__, 'sample') for i in dist_per_feature]) == len(dist_per_feature) , "Only distributions with sample function are supported for {}".format(self.__class__)
        self.dist_per_feature = dist_per_feature

    def _filter_out_of_range_samples(self, original_sample, perturbed_samples, max_distance):
        '''
        This method is used to filter out samples that are not within \epsilon norm ball.
        perturbed_samples : unfiltered samples
        '''
        original_sample_unsqueezed = original_sample.unsqueeze(0)
        distance_from_original_sample_unsqueezed = torch.cdist(perturbed_samples, original_sample_unsqueezed, p=2.0)
        perturbed_samples = perturbed_samples* (distance_from_original_sample_unsqueezed <= max_distance)
        return perturbed_samples[perturbed_samples.sum(dim = 1) != 0]


    def get_perturbed_inputs(self, original_sample : torch.FloatTensor, feature_mask : torch.BoolTensor, num_samples : int, max_distance : int) -> torch.tensor:
        '''
        feature mask : this indicates the static features
        num_samples : number of perturbed samples.
        max_distance : the maximum distance between original sample and purturbed samples.

        '''
        assert len(feature_mask) == len(
            original_sample), "mask size == original sample in get_perturbed_inputs for {}".format(self.__class__)

        perturbed_cols = []
        for i,_ in enumerate(original_sample):
            perturbed_cols.append(self.dist_per_feature[i].sample((num_samples,)).unsqueeze(1))
        perturbed_samples = original_sample + torch.cat(perturbed_cols, 1) * (~feature_mask)

        return self._filter_out_of_range_samples(original_sample, perturbed_samples, max_distance)


class AdversarialPerturbation(BasePerturbation):
    def __init__(self, data_format):
        super(UniformPerturbation, self).__init__(data_format)

    def get_perturbed_inputs(self, feature_mask, num_samples, max_distance):
        '''

        feature mask : this indicates the static features
        num_samples : number of perturbed samples.
        max_distance : the maximum distance between original sample and purturbed samples.


        '''
        perturbed_samples = []
        return perturbed_samples