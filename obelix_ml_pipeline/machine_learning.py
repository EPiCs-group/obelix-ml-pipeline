# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.express as px


# function to perform stratified shuffle split on substate_names_column, gridsearchcv and cross_val_score on model
# afterwards the best RF model and its performance is returned
def train_classifier(train_data, ligand_numbers_column, substrate_names_column, target, test_size, cv=5, scoring='balanced_accuracy', n_jobs=1, print_results=False):
    print('Training classifier')
    # v3, attempt with just using for loop
    X = train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1)
    y = train_data[target]

    # define parameters for gridsearchcv
    param_grid = {
        'bootstrap': [False],
        'max_depth': [5, 50, 100],
        'max_features': [3, 5],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [5, 10],
        'n_estimators': [50, 100]
    }

    # create a base random forest model
    rf = RandomForestClassifier(random_state=42)

    # define the stratified shuffle split
    # sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size, random_state=42)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    splits = []
    for train_index, test_index in skf.split(X, train_data[substrate_names_column]):
        splits.append((train_index, test_index))

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=splits, n_jobs=n_jobs, scoring=scoring)
    # fit the gridsearchcv object
    grid_search.fit(X, y)
    # get the best model
    best_model = grid_search.best_estimator_
    # now fit to the whole training set
    predictions = best_model.predict(X)
    # get the performance of the best model
    best_model_performance = balanced_accuracy_score(y, predictions)
    # get the confusion matrix
    best_model_confusion_matrix = confusion_matrix(y, predictions)

    # extract the mean and std of the performance of the best model across folds
    training_test_scores_mean = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
    training_test_scores_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]

    if print_results:
        print('Mean test performance: {:.2f} +/- {:.2f}'.format(training_test_scores_mean, training_test_scores_std))

    # plot confusion matrix of the best model with matplotlib
    fig_cm, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(best_model_confusion_matrix, annot=True, fmt='d', ax=ax)
    ax.set_title('Confusion matrix of the best model')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')


    # figure for the feature importance of the best model interactively with plotly sorted by importance
    feature_importances = pd.DataFrame(best_model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
    fig_fi = px.bar(feature_importances, x=feature_importances.index, y='importance', title='Feature importance of the best RF model')
    fig_fi.update_xaxes(title_text='Feature')
    fig_fi.update_yaxes(title_text='Importance')

    return best_model, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi


# function to use trained classifier to predict on test data
def predict_classifier(test_data, ligand_numbers_column, substrate_names_column, target, model, print_results=False):
    print('Using trained classifier for predictions on test set...')
    # predict on test set
    y_pred = model.predict(test_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1))
    y_true = test_data[target]

    # print performance on test set
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    if print_results:
        print('Test set performance: ', balanced_accuracy)

    # print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if print_results:
        print('Test set confusion matrix: ', cm)

    # plot confusion matrix
    fig_cm, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', linewidths=.5, ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    return fig_cm, balanced_accuracy, cm


# function for preparing data for either binary or multiclass classification
def prepare_classification_df(df, target, threshold, binary=True):
    df = df.copy()
    # Encode target variable as numerical labels
    # le = LabelEncoder()
    # df[target] = le.fit_transform(df[target])

    if binary:
        # Convert target variable to binary based on the threshold
        df.loc[df[target] < threshold, target] = 0
        df.loc[df[target] >= threshold, target] = 1
        df[target] = df[target].astype(np.int64)
    else:
        # Do nothing for multiclass classification
        pass

    return df
