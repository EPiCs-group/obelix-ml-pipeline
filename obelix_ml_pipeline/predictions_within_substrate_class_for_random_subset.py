# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

# objective 4: cluster data (KNN clustering) based on ligand properties and make predictions for each cluster
# import libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

from obelix_ml_pipeline.load_representations import select_features_for_representation, load_and_merge_representations_and_experimental_response, load_ligand_representations,load_experimental_response,merge_dfs
from obelix_ml_pipeline.machine_learning import prepare_classification_df, train_ml_model, predict_ml_model, reduce_dimensionality_train_test
from obelix_ml_pipeline.data_classes import PredictionResults


def predict_within_substrate_class_for_random_subset(selected_ligand_representations,
                                   ligand_numbers_column, substrate_names_column, target, target_threshold,
                                   train_splits, binary,
                                   selected_substrate, training_size, rf_model, scoring, print_ml_results, n_jobs,
                                   subset_random_seeds, performance_threshold,
                                   plot_dendrograms=False,
                                   reduce_train_test_data_dimensionality=False, transformer=None):
    ligand_features = [select_features_for_representation(representation_type, ligand=True) for representation_type in
                       selected_ligand_representations]
    # flatten list of lists
    ligand_features = [item for sublist in ligand_features for item in sublist]

    features = ligand_features  # in this case the model is substrate specific, so we only use ligand features

    # load selected ligand representations and experimental response
    # df = load_and_merge_representations_and_experimental_response(selected_ligand_representations, plot_dendrograms)
    selected_features = select_features_for_representation(selected_ligand_representations[0], ligand=True)
    ligand_index_column = 'Ligand#'
    selected_features_and_ligand_index = selected_features + [ligand_index_column]
    first_ligand_rep = load_ligand_representations(selected_ligand_representations[0],
                                                   columns_of_representation_to_select=selected_features_and_ligand_index)
    exp_df = load_experimental_response()
    df = merge_dfs(exp_df, ligand_index_column, first_ligand_rep, ligand_index_column)
    if len(selected_ligand_representations) > 1:
        for selected_ligand_representation in selected_ligand_representations[1:]:
            selected_features = select_features_for_representation(selected_ligand_representation, ligand=True)
            selected_features_and_ligand_index = selected_features + [ligand_index_column]
            ligand_rep = load_ligand_representations(selected_ligand_representation,
                                                     columns_of_representation_to_select=selected_features_and_ligand_index)
            # keep merging ligand rep to first ligand rep and exp df
            df = merge_dfs(df, ligand_index_column, ligand_rep, ligand_index_column)

    # for the dataframe we want the ligand number, substrate name, target and ligand/substrate features
    df = df[[ligand_numbers_column, substrate_names_column, target] + features]

    randomly_chosen_fraction = 0.9  # fraction of data that is randomly selected to create a model
    best_best_model = None
    best_training_test_scores_mean = None
    best_training_test_scores_std = None
    best_training_best_model_performance = None
    best_fig_cm = None
    best_fig_fi = None
    best_testing_confusion_fig = None
    best_testing_cm_test = None
    best_testing_performance_test = 0
    best_train_data = None
    best_test_data = None
    best_random_seed = None
    best_randomly_chosen_fraction = None

    # we keep decreasing the fraction of the data that is randomly selected to create a model until the performance
    # of the model is above the threshold or the fraction is 0.1. We're trying to see if there is a trend in
    # data for which the model performs well
    while best_testing_performance_test < performance_threshold and randomly_chosen_fraction > 0.1:
        for random_seed in subset_random_seeds:
            # print(random_seed)
            # print(randomly_chosen_fraction)
            # choose subset of the data based on substrate
            subset_data = df.loc[df[substrate_names_column] == selected_substrate]
            # choose random subset of the data based on the randomly_chosen_fraction for training/test
            subset_data = subset_data.sample(frac=randomly_chosen_fraction, random_state=random_seed)

            if binary:
                subset_train, subset_test = train_test_split(subset_data, test_size=1 - training_size, random_state=42,
                                                             stratify=subset_data[target].values)
            else:
                subset_train, subset_test = train_test_split(subset_data, test_size=1 - training_size, random_state=42)

            # in case of a binary classification task we need to transform the target column to a binary column
            if 'accuracy' in scoring:  # this means that we are doing a classification task
                if target_threshold is None:
                    target_threshold = subset_train[target].median()
                    print(f'No target threshold provided, using median of target column in training data as threshold: {target_threshold}')
                subset_train = prepare_classification_df(subset_train, target, target_threshold, binary)
                subset_test = prepare_classification_df(subset_test, target, target_threshold, binary)
            train_data = subset_train
            test_data = subset_test

            # reduce dimensionality of train and test data
            if reduce_train_test_data_dimensionality and transformer is not None:
                scaler, transformer, train_data, test_data = reduce_dimensionality_train_test(train_data, test_data, target,
                                                                                              ligand_numbers_column,
                                                                                              substrate_names_column,
                                                                                              transformer)

            best_model, training_best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi = train_ml_model(
                train_data, ligand_numbers_column, substrate_names_column, target, rf_model, train_splits, scoring, n_jobs, print_ml_results)

            testing_performance_test, testing_confusion_fig, testing_cm_test = predict_ml_model(test_data,
                                                                                        ligand_numbers_column,
                                                                                        substrate_names_column, target,
                                                                                        best_model, scoring=scoring,
                                                                                        print_results=print_ml_results)

            if testing_performance_test > best_testing_performance_test:
                best_best_model = best_model
                best_training_test_scores_mean = training_test_scores_mean
                best_training_test_scores_std = training_test_scores_std
                best_training_best_model_performance = training_best_model_performance
                best_fig_cm = fig_cm
                best_fig_fi = fig_fi
                best_testing_performance_test = testing_performance_test
                best_testing_confusion_fig = testing_confusion_fig
                best_testing_cm_test = testing_cm_test
                best_train_data = train_data
                best_test_data = test_data
                best_random_seed = random_seed
                best_randomly_chosen_fraction = randomly_chosen_fraction

        randomly_chosen_fraction -= 0.1
    prediction_results = PredictionResults(best_best_model, best_training_best_model_performance, best_training_test_scores_mean,
                                             best_training_test_scores_std, best_fig_cm, best_fig_fi, best_testing_performance_test,
                                                best_testing_confusion_fig, best_testing_cm_test, best_train_data, best_test_data, best_random_seed, best_randomly_chosen_fraction)
    return prediction_results


if __name__ == "__main__":
    # try classifier with loaded representations
    selected_ligand_representations = ['dft_nbd_model']
    target = 'DDG'
    target_threshold = None
    rf_model = RandomForestRegressor(random_state=42)
    scoring = 'r2'
    train_splits = 5
    n_jobs = 4
    binary = False
    plot_dendrograms = False
    substrate_names_column = 'Substrate'
    ligand_numbers_column = 'Ligand#'
    selected_substrate = 'SM1'
    training_size = 0.8  # 80% of data for subset substrate is used for training
    print_ml_results = True
    reduce_train_test_data_dimensionality = False
    transformer = None
    random_seeds = np.arange(0, 10, 1)
    performance_threshold = 0.75
    print('Training and testing classifier')
    print(f'Test size in training (based on K-fold): {1 / train_splits}')
    prediction_results = predict_within_substrate_class_for_random_subset(selected_ligand_representations,
                                   ligand_numbers_column, substrate_names_column, target, target_threshold,
                                   train_splits, binary,
                                   selected_substrate, training_size, rf_model, scoring, print_ml_results, n_jobs,
                                   random_seeds, performance_threshold, plot_dendrograms,
                                   reduce_train_test_data_dimensionality, transformer)

