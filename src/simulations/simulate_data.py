import numpy as np
import pandas as pd

import src.simulations.config as config
from src.utils.common import black_scholes_price
from sklearn.model_selection import train_test_split

class DataGenerator():
    def __init__(self, 
                 param_bounds=None, 
                 sample_sizes=None,
                 strikes_per_S=None,
                 gen_seed=None,
                 split_seed=None,
                 split=True):
        
        """
        Generate synthetic data for the option pricing problem.

        Parameters
        ----------
        param_bounds : dict
            Dictionary with the bounds for the parameters S, sigma, tau and r.
            Default is None, which uses the values from the `src.simulations.config` file.

        sample_sizes : list or int
            List of integers with the number of samples to generate.
            Default is None, which uses the values from the `src.simulations.config` file.
        
        strikes_per_S : int
            Number of strikes to generate for each value of S. 
            A unique point of {S, sigma, tau, r} will be generated, and then `strikes_per_S` number 
            of times it is copied with different K values sampled around S. 
            Default is None, which uses the value from the `src.simulations.config` file.
        
        gen_seed : int
            Seed for the generation of the synthetic data.

        split_seed : int
            Seed for the train-test spliting, and proper training - calibration spliting.
        
        split : bool
            If True, the data will be split into train, proper train, calibration and validation sets.
            If False, the data will not be split and will be returned as a single dataframe.
        """

        if param_bounds is None:
            self.param_bounds = config.param_bounds
        else:
            self.param_bounds = param_bounds

        if sample_sizes is None:
            self.sample_sizes = config.sample_sizes
        else:
            self.sample_sizes = sample_sizes

        if strikes_per_S is None:
            self.strikes_per_S = config.strikes_per_S
        else:
            self.strikes_per_S = strikes_per_S

        if gen_seed is None:
            self.gen_seed = config.gen_seed
        else:
            self.gen_seed = gen_seed

        if split_seed is None:
            self.split_seed = config.split_seed
        else:
            self.split_seed = split_seed
        
        self.split = split

    def _make_single_dataset(self, num_independent_samples):
        # generate data from uniform distribution for each parameter
        np.random.seed(self.gen_seed)
        data = {}
        for param in self.param_bounds.keys():
            data[param] = np.repeat( # repeat the same sampled value strikes_per_S times
                np.random.uniform(
                    self.param_bounds[param][0], # lower bound
                    self.param_bounds[param][1], # upper bound 
                    num_independent_samples), 
                self.strikes_per_S)

        # generate 'strikes_per_S' number of strikes from normal with mean S and std 0.1*S, for each S
        z = np.random.normal(
                1, # mean
                # variance of 0.1 => std=sqrt(0.1)=0.316
                # ~300 strikes will be negative
                # the author probably used 0.1 as a variance, not std.
                # With std=0.1, the number of negative strikes will be ~0
                0.1, 
                num_independent_samples*self.strikes_per_S)

        data['K'] = data['S']/z

        data = pd.DataFrame(data)

        # Add the option price
        data['C'] = black_scholes_price(data['S'], data['K'], data['r'], data['sigma'], data['tau'], option_type="call")
        
        # Scale prices with K
        data['S'] = data['S']/data['K']
        data['C'] = data['C']/data['K']
        data.drop('K', axis=1, inplace=True)

        return data


    def generate_datasets(self, sample_sizes=None, gen_seed=None, split_seed=None, split=False):
        if sample_sizes is not None:
            self.sample_sizes = sample_sizes
        if gen_seed is not None:
            self.gen_seed = gen_seed
        if split_seed is not None:
            self.split_seed = split_seed

        if type(self.sample_sizes) is not list:
            self.sample_sizes = [self.sample_sizes]

        num_independent_samples = int(sum(self.sample_sizes)/self.strikes_per_S)

        data = self._make_single_dataset(num_independent_samples)

        data['sample_id'] = 0
        # Label the samples with the sample id, according to the sample_sizes list.
        # This will make it easier to split dataframe into samples of different sizes.
        # If sample_sizes was a single number, all samples would have the same id of 0.
        
        for i, _ in enumerate(self.sample_sizes):
            data.loc[data.index.isin(range(sum(self.sample_sizes[:i]), sum(self.sample_sizes[:i+1]))), 'sample_id'] = i

            sample_ids = data['sample_id'].unique()
            samples = [data.loc[data['sample_id']==id] for id in sample_ids]
            sample_lens = [len(sample) for sample in samples]

            data_sets = {}

            for sample in samples:
                id = sample['sample_id'].iloc[0]
                size = len(sample)
                sample = sample.drop('sample_id', axis=1)

                if split:
                    # Split into samples by sample_id
                    train, validation = train_test_split(sample, test_size=0.2, random_state=self.split_seed)
                    proper_train, calibration = train_test_split(train, test_size=0.25, random_state=self.split_seed) # 0.25 x 0.8 = 0.2
                    data_sets[id] = [train, proper_train, calibration, validation]
                else:
                    data_sets[id] = sample

        return data_sets
