# Script for hyperparameter optimization on simulation data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.metrics import mean_absolute_error
import time
import pickle

data = pd.read_csv('data/simulated/simulation_1.csv')

############## Settings ###################

# Define grid search parameters from table 2
param_grid = {
    'n_estimators': [100, 500, 1000, 2500, 5000],
    'num_leaves': [32, 64, 128, 256],
    'max_depth': [8, 16, 32, 64],
    'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5]
}

# Train models for conformal prediction (use 'proper_train' set) or non-cp (use 'train' set)
# type_of_training = 'proper_train' # 'train'

############################################

# Assume homogeneity
data['S'] = data['S']/data['K']
data['C'] = data['C']/data['K']
data.drop('K', axis=1, inplace=True)

# Split into samples by sample_id
sample_ids = data['sample_id'].unique()
samples = [data.loc[data['sample_id']==id] for id in sample_ids]
sample_lens = [len(sample) for sample in samples]

# Split into training and testing sets, for each sample 
tr_val_sets = {}

start_time = time.time()

for type_of_training in ['proper_train', 'train']:

    for sample in samples:
        id = sample['sample_id'].iloc[0]
        size = len(sample)
        sample = sample.drop('sample_id', axis=1)

        train, validation = train_test_split(sample, test_size=0.2, random_state=42)

        if type_of_training == 'proper_train':
            proper_train, _ = train_test_split(train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2
            tr_val_sets[id] = [proper_train, validation]
        else:
            tr_val_sets[id] = [train, validation]

    results_per_sample = {}

    for id in sample_ids:
        train, validation = tr_val_sets[id]

        # make a dataframe to store the results
        results = pd.DataFrame(columns=['n_estimators', 'num_leaves', 'max_depth', 'learning_rate', 'mae'])

        param_progress_counter = 0
        # Perform grid search

        min_mae = float('inf')
        best_model = None

        for params in ParameterGrid(param_grid):
            param_progress_counter += 1
            print(f"Set: {type_of_training};  Sample:{id}/5;  Grid:{param_progress_counter/4}%")
            # Train the model with the current set of hyperparameters
            model = lgb.LGBMRegressor(objective='quantile', alpha=0.5, **params, n_jobs=7, verbose=-1)

            model.fit(train.drop('C', axis=1), train['C'])

            # Predict on the validation set
            y_pred = model.predict(validation.drop('C', axis=1))

            # Calculate Mean Absolute Error
            mae = mean_absolute_error(validation['C'], y_pred)

            results = pd.concat([results, pd.DataFrame({'n_estimators': params['n_estimators'],
                            'num_leaves': params['num_leaves'],
                            'max_depth': params['max_depth'],
                            'learning_rate': params['learning_rate'],
                            'mae': mae}, index=[0])])
            
            # Keep the best model
            if mae < min_mae:
                min_mae = mae
                best_model = model

        # Save the best model as a pickle file
        pickle.dump(best_model, open(f"models/simulation_1/sample_{id}/model_{type_of_training}.pkl", 'wb'))
        results.to_csv(f"models/simulation_1/sample_{id}/results_{type_of_training}.csv", index=False)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
