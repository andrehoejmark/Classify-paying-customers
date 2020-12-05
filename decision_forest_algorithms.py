import pandas as pd
import numpy as np

from decision_forest_functions import decision_tree_algorithm, decision_tree_predictions
from framework_utils import forest_train_test_split, calculate_accuracy


def forest_predict(data_frame_scaled):
    # We predict Revenue
    revenue_val = data_frame_scaled.Revenue.value_counts(normalize=True)
    revenue_val = revenue_val.sort_index()
    revenue_val.plot(kind="bar")

    # From here Revenue gets another name because that is  how the forest algorithm is implemented
    # so for other algorithms take the data before this step
    data_frame_scaled["label"] = data_frame_scaled.Revenue
    data_frame_scaled = data_frame_scaled.drop("Revenue", axis=1)
    train_df, test_df = forest_train_test_split(data_frame_scaled, 0.25)
    forest = random_forest_algorithm(train_df, n_trees=5, n_bootstrap=1200, n_features=2, dt_max_depth=5)
    predictions = random_forest_predictions(test_df, forest)
    accuracy = calculate_accuracy(predictions, test_df.label)

    print("Accuracy Random Forest = {}".format(accuracy))



def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]

    return df_bootstrapped


def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)

    return forest


def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]

    return random_forest_predictions
