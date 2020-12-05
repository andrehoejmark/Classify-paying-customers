# coding: utf-8

import random

import numpy as np

"""
 take the indices of the rows numbers from data_frame
 then shuflle; than generate the random numbers from the probability distribution 
 the random numbers represent the indices of elements obtained after the shuflling step
"""


# 1. Train-Test-Split
def forest_train_test_split(data_frame, test_proportion=0.25):
    test_size = 0
    if isinstance(test_proportion, float):
        test_size = round(test_proportion * len(data_frame))

    # data_size = data_frame.length
    indices = data_frame.index.tolist()

    # randomly shuffle 
    from numpy.random import seed
    from numpy.random import shuffle
    # seed random number generator
    seed(1)
    # data indices
    indices = len(indices)
    # randomly shuffle the data indices
    #shuffle(indices)

    test_indices = np.random.randint(1, high=indices + 1, size=test_size)

    test_df = data_frame.loc[test_indices]
    train_df = data_frame.drop(test_indices)

    return train_df, test_df


# backup function
def forest_train_test_split_simple(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


# 2. Distinguish categorical and continuous features
def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_threshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_threshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")

    return feature_types


# 3. Accuracy
def calculate_accuracy(predictions, labels):
    predictions_correct = predictions == labels
    accuracy = predictions_correct.mean()

    return accuracy
