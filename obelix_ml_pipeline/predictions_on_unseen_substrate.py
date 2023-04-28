# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

# objective 1: For a complete new substrate, the model gives the performance of 192 ligands with an accuracy as high as possible
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from obelix_ml_pipeline.load_representations import prepare_selected_representations_df
from obelix_ml_pipeline.machine_learning import prepare_classification_df, train_ml_model, predict_ml_model, reduce_dimensionality_train_test
from obelix_ml_pipeline.data_classes import PredictionResults


def predict_out_of_sample_substrate(selected_ligand_representations, selected_substrate_representations,
                                    ligand_numbers_column, substrate_names_column, target, target_threshold, train_splits, binary,
                                    list_of_training_substrates, list_of_test_substrates, rf_model, scoring, print_ml_results, n_jobs, plot_dendrograms=False,
                                    reduce_train_test_data_dimensionality=False, transformer=None):
    df = prepare_selected_representations_df(selected_ligand_representations, selected_substrate_representations, ligand_numbers_column, substrate_names_column, target, plot_dendrograms)

    train_data = df.copy()
    train_data = train_data[train_data[substrate_names_column].isin(list_of_training_substrates)]
    test_data = df.copy()
    test_data = test_data[test_data[substrate_names_column].isin(list_of_test_substrates)]

    # in case of a binary classification task we need to transform the target column to a binary column
    if 'accuracy' in scoring:  # this means that we are doing a classification task
        if target_threshold is None:
            target_threshold = train_data[target].median()
            print(f'No target threshold provided, using median of target column in training data as threshold: {target_threshold}')
        train_data = prepare_classification_df(train_data, target, target_threshold, binary)
        test_data = prepare_classification_df(test_data, target, target_threshold, binary)

    # reduce dimensionality of train and test data
    if reduce_train_test_data_dimensionality and transformer is not None:
        scaler, transformer, train_data, test_data = reduce_dimensionality_train_test(train_data, test_data, target, ligand_numbers_column, substrate_names_column, transformer)

    best_model, training_best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi = train_ml_model(
        train_data, ligand_numbers_column, substrate_names_column,
        target,
        rf_model=rf_model, cv=train_splits, scoring=scoring, n_jobs=n_jobs,
        print_results=print_ml_results)

    # # test model on test set
    testing_performance_test, testing_confusion_fig, testing_cm_test = predict_ml_model(test_data, ligand_numbers_column,
                                                                                        substrate_names_column, target,
                                                                                        best_model, scoring=scoring,
                                                                                        print_results=print_ml_results)

    # load results in a class
    prediction_results = PredictionResults(best_model, training_best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi, testing_performance_test, testing_confusion_fig, testing_cm_test)

    return prediction_results


if __name__ == "__main__":
    from sklearn.decomposition import PCA
    # try classifier with loaded representations
    selected_ligand_representations = ['dft_nbd_model_with_solvation']
    selected_substrate_representations = ['sterimol']
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
    list_of_training_substrates = ['SM1', 'SM2']
    list_of_test_substrates = ['SM3']
    print_ml_results = True
    reducte_train_test_data_dimensionality = True
    transformer = PCA(n_components=0.95, random_state=42)
    print('Training and testing classifier')
    print(f'Test size in training (based on K-fold): {1/train_splits}')
    # do the same with general function predict_out_of_sample_substrate
    # load all output in a class
    prediction_results = predict_out_of_sample_substrate(
        selected_ligand_representations, selected_substrate_representations, ligand_numbers_column,
        substrate_names_column, target, target_threshold, train_splits, binary=binary,
        list_of_training_substrates=list_of_training_substrates, list_of_test_substrates=list_of_test_substrates,
        rf_model=rf_model, scoring=scoring, print_ml_results=print_ml_results, n_jobs=n_jobs,
        plot_dendrograms=plot_dendrograms, reduce_train_test_data_dimensionality=reducte_train_test_data_dimensionality,
        transformer=transformer)
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
    print(f'Test size: {1/train_splits}')
    prediction_results = predict_out_of_sample_substrate(
        selected_ligand_representations, selected_substrate_representations, ligand_numbers_column,
        substrate_names_column, target, target_threshold, train_splits, binary=binary,
        list_of_training_substrates=list_of_training_substrates, list_of_test_substrates=list_of_test_substrates,
        rf_model=rf_model, scoring=scoring, print_ml_results=print_ml_results, n_jobs=n_jobs, plot_dendrograms=plot_dendrograms)
    prediction_results.testing_confusion_fig.show()
