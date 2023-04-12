# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

# objective 1: For a complete new substrate, the model gives the performance of 192 ligands with an accuracy as high as possible
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from obelix_ml_pipeline.load_representations import select_features_for_representation, load_and_merge_representations_and_experimental_response
from obelix_ml_pipeline.machine_learning import prepare_classification_df, train_ml_model, predict_ml_model


def predict_out_of_sample_substrate(selected_ligand_representations, selected_substrate_representations,
                                    ligand_numbers_column, substrate_names_column, target, target_threshold, train_splits, binary,
                                    list_of_training_substrates, list_of_test_substrates, rf_model, scoring, print_ml_results, n_jobs, plot_dendrograms=False):
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
    # for the classification dataframe we want the ligand number, substrate name, target and ligand/substrate features
    df = df[[ligand_numbers_column, substrate_names_column, target] + features]
    if 'accuracy' in scoring:  # this means that we are doing a classification task
        df = prepare_classification_df(df, target, target_threshold, binary)
    train_data = df.copy()
    train_data = train_data[train_data[substrate_names_column].isin(list_of_training_substrates)]
    best_model, training_best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi = train_ml_model(
        train_data, ligand_numbers_column, substrate_names_column,
        target,
        rf_model=rf_model, cv=train_splits, scoring=scoring, n_jobs=n_jobs,
        print_results=print_ml_results)

    # # test model on test set
    test_data = df.copy()
    test_data = test_data[test_data[substrate_names_column].isin(list_of_test_substrates)]
    testing_performance_test, testing_confusion_fig, testing_cm_test = predict_ml_model(test_data, ligand_numbers_column,
                                                                                        substrate_names_column, target,
                                                                                        best_model, scoring=scoring,
                                                                                        print_results=print_ml_results)
    return best_model, training_best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi, testing_performance_test, testing_confusion_fig, testing_cm_test


if __name__ == "__main__":
    # choose 2 training substrates and 1 test substrate and try regression with loaded representations
    list_of_substrates = ['SM1', 'SM2', 'SM3']
    list_of_training_substrates = []
    list_of_test_substrates = []

    for test_substrate in list_of_substrates:
        list_of_training_substrates = [x for x in list_of_substrates if x != test_substrate]
        list_of_test_substrates = [test_substrate]
        print(f'Training substrates: {list_of_training_substrates}')
        print(f'Test substrates: {list_of_test_substrates}')
        selected_ligand_representations = ['dft_nbd_model']
        selected_substrate_representations = ['dft_steric_fingerprint']
        target = 'Conversion'
        target_threshold = 0.8
        rf_model = RandomForestClassifier(random_state=42)
        scoring = 'balanced_accuracy'
        train_splits = 5
        n_jobs = 1
        binary = True
        plot_dendrograms = False
        substrate_names_column = 'Substrate'
        ligand_numbers_column = 'Ligand#'
        list_of_training_substrates = ['SM1', 'SM2']
        list_of_test_substrates = ['SM3']
        print_ml_results = False

        print('Training and testing classification')
        print(f'Test size: {1 / train_splits}')

        print(f'Testing substrate representation: {selected_substrate_representations}')
        best_model, best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi, testing_balanced_accuracy_test, testing_confusion_fig, testing_cm_test = predict_out_of_sample_substrate(
            selected_ligand_representations, selected_substrate_representations, ligand_numbers_column,
            substrate_names_column, target, target_threshold, train_splits, binary=binary,
            list_of_training_substrates=list_of_training_substrates, list_of_test_substrates=list_of_test_substrates,
            rf_model=rf_model, scoring=scoring, print_ml_results=print_ml_results, n_jobs=n_jobs,
            plot_dendrograms=plot_dendrograms)
        print('Mean test performance during training: {:.2f} +/- {:.2f}'.format(training_test_scores_mean,
                                                                                training_test_scores_std))
        # print(f'Confusion matrix: {testing_cm_test}')
        print(f'Balanced accuracy on test substrate: {testing_balanced_accuracy_test}')
        # print(f'Feature importance: {fig_fi}')
        # print(f'Confusion matrix: {fig_cm}')
        print('----------------------------------------')

    # # try classifier with loaded representations
    # selected_ligand_representations = ['dft_nbd_model']
    # selected_substrate_representations = ['sterimol']
    # target = 'Conversion'
    # target_threshold = 0.8
    # rf_model = RandomForestClassifier(random_state=42)
    # scoring = 'balanced_accuracy'
    # train_splits = 5
    # n_jobs = 4
    # binary = True
    # plot_dendrograms = False
    # substrate_names_column = 'Substrate'
    # ligand_numbers_column = 'Ligand#'
    # list_of_training_substrates = ['SM1', 'SM2']
    # list_of_test_substrates = ['SM3']
    # print_ml_results = True
    # print('Training and testing classifier')
    # print(f'Test size in training (based on K-fold): {1/train_splits}')
    # # do the same with general function predict_out_of_sample_substrate
    # best_model, best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi, \
    #     testing_balanced_accuracy_test, testing_confusion_fig,  testing_cm_test = predict_out_of_sample_substrate(
    #     selected_ligand_representations, selected_substrate_representations, ligand_numbers_column,
    #     substrate_names_column, target, target_threshold, train_splits, binary=binary,
    #     list_of_training_substrates=list_of_training_substrates, list_of_test_substrates=list_of_test_substrates,
    #     rf_model=rf_model, scoring=scoring, print_ml_results=print_ml_results, n_jobs=n_jobs,
    #     plot_dendrograms=plot_dendrograms)
    # fig_cm.show()
    # # fig_fi.show()
    # testing_confusion_fig.show()
    #
    # # try regression with loaded representations
    # target = 'EE'
    # target_threshold = 0.6
    # rf_model = RandomForestRegressor(random_state=42)
    # scoring = 'r2'
    # binary = False
    # print('Training and testing regression')
    # print(f'Test size: {1/train_splits}')
    # best_model, best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi, testing_balanced_accuracy_test, testing_confusion_fig, testing_cm_test = predict_out_of_sample_substrate(
    #     selected_ligand_representations, selected_substrate_representations, ligand_numbers_column,
    #     substrate_names_column, target, target_threshold, train_splits, binary=binary,
    #     list_of_training_substrates=list_of_training_substrates, list_of_test_substrates=list_of_test_substrates,
    #     rf_model=rf_model, scoring=scoring, print_ml_results=print_ml_results, n_jobs=n_jobs, plot_dendrograms=plot_dendrograms)
    # testing_confusion_fig.show()
