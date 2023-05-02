# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

# objective 3: Within a substrate class, the model gives the performance of 192 ligands with an accuracy as high as possible
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


from obelix_ml_pipeline.load_representations import select_features_for_representation, load_and_merge_representations_and_experimental_response, load_ligand_representations,load_experimental_response,merge_dfs
from obelix_ml_pipeline.machine_learning import prepare_classification_df, train_ml_model, predict_ml_model, reduce_dimensionality_train_test
from obelix_ml_pipeline.data_classes import PredictionResults

def predict_within_substrate_class(selected_ligand_representations,
                                    ligand_numbers_column, substrate_names_column, target, target_threshold, train_splits, binary,
                                    selected_substrate, training_size, rf_model, scoring, print_ml_results, n_jobs, plot_dendrograms=False,
                                   reduce_train_test_data_dimensionality=False, transformer=None):
    
    ligand_features = [select_features_for_representation(representation_type, ligand=True) for representation_type in
                       selected_ligand_representations]
    # flatten list of lists
    ligand_features = [item for sublist in ligand_features for item in sublist]

    features = ligand_features 

    # load selected representations and experimental response
    #df = load_and_merge_representations_and_experimental_response(selected_ligand_representations, plot_dendrograms)
    selected_features = select_features_for_representation(selected_ligand_representations[0], ligand=True)
    ligand_index_column = 'Ligand#'
    selected_features_and_ligand_index = selected_features + [ligand_index_column]
    first_ligand_rep = load_ligand_representations(selected_ligand_representations[0], columns_of_representation_to_select=selected_features_and_ligand_index)
    exp_df = load_experimental_response()
    df = merge_dfs(exp_df, ligand_index_column, first_ligand_rep, ligand_index_column)
    if len(selected_ligand_representations) > 1:
        for selected_ligand_representation in selected_ligand_representations[1:]:
            selected_features = select_features_for_representation(selected_ligand_representation, ligand=True)
            selected_features_and_ligand_index = selected_features + [ligand_index_column]
            ligand_rep = load_ligand_representations(selected_ligand_representation, columns_of_representation_to_select=selected_features_and_ligand_index)
            # keep merging ligand rep to first ligand rep and exp df
            df = merge_dfs(df, ligand_index_column, ligand_rep, ligand_index_column)
            
    # for the dataframe we want the ligand number, substrate name, target and ligand/substrate features
    df = df[[ligand_numbers_column, substrate_names_column, target] + features]

    subset_data = df.loc[df[substrate_names_column] == selected_substrate]

    # old approach from when binary preparation was done on the whole dataset without dynamic target_threshold
    # if binary:
    #     subset_train, subset_test = train_test_split(subset_data, test_size=1 - training_size, random_state=42,
    #                                                  stratify=subset_data[target].values)
    # else:
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
        scaler, transformer, train_data, test_data = reduce_dimensionality_train_test(train_data, test_data, target, ligand_numbers_column, substrate_names_column, transformer)
    
    best_model, training_best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi = train_ml_model(
        train_data, ligand_numbers_column, substrate_names_column,
        target,
        rf_model=rf_model, cv=train_splits, scoring=scoring, n_jobs=n_jobs,
        print_results=print_ml_results)

    # # test model on test set
    
    testing_performance_test, testing_confusion_fig, testing_cm_test = predict_ml_model(test_data,
                                                                                        ligand_numbers_column,
                                                                                        substrate_names_column, target,
                                                                                        best_model, scoring=scoring,
                                                                                        print_results=print_ml_results)

    prediction_results = PredictionResults(best_model, training_best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi, testing_performance_test, testing_confusion_fig, testing_cm_test, train_data, test_data)
    return prediction_results


if __name__ == '__main__':
    # try classifier with loaded representations
    selected_ligand_representations = ['dft_nbd_model']
    target = 'Conversion'
    target_threshold = 0.8
    rf_model = RandomForestClassifier(random_state=42)
    scoring = 'balanced_accuracy'
    train_splits = 5
    n_jobs = 4
    binary = True
    plot_dendrograms = False
    substrate_names_column = 'Substrate'
    ligand_numbers_column = 'Ligand#'
    selected_substrate = 'SM1'
    training_size = 0.8  # 80% of data for subset substrate is used for training
    print_ml_results = True
    print('Training and testing classifier')
    print(f'Test size in training (based on K-fold): {1 / train_splits}')
    prediction_results = predict_within_substrate_class(selected_ligand_representations,
                                    ligand_numbers_column, substrate_names_column, target, target_threshold, train_splits, binary,
                                    selected_substrate, training_size, rf_model, scoring, print_ml_results, n_jobs, plot_dendrograms)
    prediction_results.fig_cm.show()
    # fig_fi.show()
    prediction_results.testing_confusion_fig.show()

    # try regression with loaded representations
    target = 'EE'
    target_threshold = 0.6
    rf_model = RandomForestRegressor(random_state=42)
    scoring = 'r2'
    binary = False
    print('Training and testing regression')
    print(f'Test size: {1 / train_splits}')
    prediction_results = predict_within_substrate_class(
        selected_ligand_representations,
        ligand_numbers_column, substrate_names_column, target, target_threshold, train_splits, binary,
        selected_substrate, training_size, rf_model, scoring, print_ml_results, n_jobs, plot_dendrograms)
    prediction_results.testing_confusion_fig.show()