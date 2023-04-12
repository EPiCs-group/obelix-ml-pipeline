# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

# objective 3: Within a substrate class, the model gives the performance of 192 ligands with an accuracy as high as possible
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


from obelix_ml_pipeline.load_representations import select_features_for_representation, load_and_merge_representations_and_experimental_response
from obelix_ml_pipeline.machine_learning import prepare_classification_df, train_ml_model, predict_ml_model


def predict_within_substrate_class(selected_ligand_representations, selected_substrate_representations,
                                    ligand_numbers_column, substrate_names_column, target, target_threshold, train_splits, binary,
                                    selected_substrate, training_size, rf_model, scoring, print_ml_results, n_jobs, plot_dendrograms=False):
    ligand_features = [select_features_for_representation(representation_type, ligand=True) for representation_type in
                       selected_ligand_representations]
    # flatten list of lists
    ligand_features = [item for sublist in ligand_features for item in sublist]
    substrate_features = [select_features_for_representation(representation_type, ligand=False) for representation_type
                          in selected_substrate_representations]
    # flatten list of lists
    substrate_features = [item for sublist in substrate_features for item in sublist]
    # print(substrate_features)
    features = ligand_features + substrate_features

    # load selected representations and experimental response
    df = load_and_merge_representations_and_experimental_response(selected_ligand_representations,
                                                                  selected_substrate_representations, plot_dendrograms)
    # for the dataframe we want the ligand number, substrate name, target and ligand/substrate features
    df = df[[ligand_numbers_column, substrate_names_column, target] + features]
    if 'accuracy' in scoring:  # this means that we are doing a classification task
        df = prepare_classification_df(df, target, target_threshold, binary)

    subset_data = df.loc[df[substrate_names_column] == selected_substrate]
    subset_train, subset_test = train_test_split(subset_data, test_size=1 - training_size, random_state=42)
    train_data = subset_train
    best_model, training_best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi = train_ml_model(
        train_data, ligand_numbers_column, substrate_names_column,
        target,
        rf_model=rf_model, cv=train_splits, scoring=scoring, n_jobs=n_jobs,
        print_results=print_ml_results)

    # # test model on test set
    test_data = subset_test
    testing_performance_test, testing_confusion_fig, testing_cm_test = predict_ml_model(test_data,
                                                                                        ligand_numbers_column,
                                                                                        substrate_names_column, target,
                                                                                        best_model, scoring=scoring,
                                                                                        print_results=print_ml_results)
    return best_model, training_best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi, testing_performance_test, testing_confusion_fig, testing_cm_test


if __name__ == '__main__':
    # try classifier with loaded representations
    selected_ligand_representations = ['dft_nbd_model']
    selected_substrate_representations = ['dft_steric_fingerprint']
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
    best_model, training_best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi, testing_performance_test, testing_confusion_fig, testing_cm_test = predict_within_substrate_class(selected_ligand_representations, selected_substrate_representations,
                                    ligand_numbers_column, substrate_names_column, target, target_threshold, train_splits, binary,
                                    selected_substrate, training_size, rf_model, scoring, print_ml_results, n_jobs, plot_dendrograms)
    fig_cm.show()
    # fig_fi.show()
    testing_confusion_fig.show()

    # try regression with loaded representations
    target = 'EE'
    target_threshold = 0.6
    rf_model = RandomForestRegressor(random_state=42)
    scoring = 'r2'
    binary = False
    print('Training and testing regression')
    print(f'Test size: {1 / train_splits}')
    best_model, training_best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi, testing_performance_test, testing_confusion_fig, testing_cm_test = predict_within_substrate_class(
        selected_ligand_representations, selected_substrate_representations,
        ligand_numbers_column, substrate_names_column, target, target_threshold, train_splits, binary,
        selected_substrate, training_size, rf_model, scoring, print_ml_results, n_jobs, plot_dendrograms)
    testing_confusion_fig.show()