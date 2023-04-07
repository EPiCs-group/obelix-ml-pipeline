# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

# objective 1: For a complete new substrate, the model gives the performance of 192 ligands with an accuracy as high as possible
from obelix_ml_pipeline.load_representations import select_features_for_representation, load_and_merge_representations_and_experimental_response
from obelix_ml_pipeline.machine_learning import prepare_classification_df, train_classifier, predict_classifier


def predict_out_of_sample_substrate_classification(selected_ligand_representations, selected_substrate_representations,
                                    ligand_numbers_column, substrate_names_column, target, target_threshold, train_splits, binary,
                                    list_of_training_substrates, list_of_test_substrates, test_size, print_ml_results, n_jobs):
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
    #
    # load selected representations and experimental response
    df = load_and_merge_representations_and_experimental_response(selected_ligand_representations,
                                                                  selected_substrate_representations)
    # for the classification dataframe we want the ligand number, substrate name, target and ligand/substrate features
    classification_df = df[[ligand_numbers_column, substrate_names_column, target] + features]
    classification_df = prepare_classification_df(classification_df, target, target_threshold, binary)
    train_data_classification = classification_df[
        classification_df[substrate_names_column].isin(list_of_training_substrates)]
    best_model, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi = train_classifier(train_data_classification, ligand_numbers_column, substrate_names_column,
                                          target,
                                          test_size=test_size, cv=train_splits, scoring='balanced_accuracy', n_jobs=n_jobs,
                                          print_results=print_ml_results)

    # # test model on test set
    test_data_classification = classification_df[classification_df[substrate_names_column].isin(list_of_test_substrates)]
    testing_confusion_fig, testing_balanced_accuracy_test, testing_cm_test = predict_classifier(test_data_classification,
                                                                                       ligand_numbers_column,
                                                                                       substrate_names_column, target,
                                                                                       best_model,
                                                                                       print_results=print_ml_results)
    return best_model, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi, testing_confusion_fig, testing_balanced_accuracy_test, testing_cm_test


if __name__ == "__main__":
    # try classifier with loaded representations
    selected_ligand_representations = ['dft_nbd_model']
    selected_substrate_representations = ['ecfp']
    target = 'Conversion'
    target_threshold = 0.8
    test_size = 0.2
    train_splits = 5
    n_jobs = 4
    binary = True
    plot_dendrograms = False
    substrate_names_column = 'Substrate'
    ligand_numbers_column = 'Ligand#'
    list_of_training_substrates = ['SM1', 'SM2']
    list_of_test_substrates = ['SM3']
    print_ml_results = True
    best_model, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi, testing_confusion_fig, testing_balanced_accuracy_test, testing_cm_test = predict_out_of_sample_substrate_classification(selected_ligand_representations, selected_substrate_representations, ligand_numbers_column, substrate_names_column, target, target_threshold, train_splits, binary=binary, list_of_training_substrates=list_of_training_substrates, list_of_test_substrates=list_of_test_substrates, test_size=test_size, print_ml_results=print_ml_results, n_jobs=n_jobs)
    fig_cm.show()
    fig_fi.show()
    testing_confusion_fig.show()
