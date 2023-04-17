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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.express as px


def train_ml_model(train_data, ligand_numbers_column, substrate_names_column, target, rf_model, cv, scoring, n_jobs, print_results=False):
    # define features and target
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

    # create a base random forest model (can be regression or classification based on input)
    rf = rf_model

    # define the stratified shuffle split or stratified k-fold cross validation
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
    if 'accuracy' in scoring:  # in this case we're dealing with classification
        # get the performance of the best model
        best_model_performance = balanced_accuracy_score(y, predictions)
        # get the confusion matrix
        best_model_confusion_matrix = confusion_matrix(y, predictions)
        # plot confusion matrix of the best model with matplotlib
        fig_cm, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.heatmap(best_model_confusion_matrix, annot=True, fmt='d', ax=ax)
        ax.set_title('Confusion matrix of the best model')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    else:  # in this case we're dealing with regression
        best_model_performance = r2_score(y, predictions)
        fig_cm = None

    # extract the mean and std of the performance of the best model across folds
    training_test_scores_mean = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
    training_test_scores_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]

    if print_results:
        print('Mean test performance: {:.2f} +/- {:.2f}'.format(training_test_scores_mean, training_test_scores_std))
        print('Best model performance: {:.2f}'.format(best_model_performance))
        print('Best model parameters: {}'.format(grid_search.best_params_))

    # figure for the feature importance of the best model interactively with plotly sorted by importance
    feature_importances = pd.DataFrame(best_model.feature_importances_, index=X.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    fig_fi = px.bar(feature_importances, x=feature_importances.index, y='importance',
                    title='Feature importance of the best RF model')
    fig_fi.update_xaxes(title_text='Feature')
    fig_fi.update_yaxes(title_text='Importance')

    return best_model, best_model_performance, training_test_scores_mean, training_test_scores_std, fig_cm, fig_fi


def predict_ml_model(test_data, ligand_numbers_column, substrate_names_column, target, model, scoring, print_results):
    # predict on test set
    y_pred = model.predict(test_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1))
    y_true = test_data[target]

    if 'accuracy' in scoring:  # for classification we use balanced accuracy and confusion matrix
        performance = balanced_accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
    else:  # for regression we use r2 score
        performance = r2_score(y_true, y_pred)
        cm = None

    if print_results:
        print(f'Test set performance {scoring}: ', performance)
        if 'accuracy' in scoring:
            print('Test set confusion matrix: ', cm)

    fig_cm = None
    if 'accuracy' in scoring and cm is not None:
        # plot confusion matrix
        fig_cm, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', linewidths=.5, ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion matrix of the best model')
    if 'r2' in scoring and cm is None:
        # then we're dealing with regresssion and  can make a scatter plot of the true vs predicted values
        fig_cm, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(y_true, y_pred, s=10)
        # also plot the regression line and display the r2 score
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4)
        ax.text(0.05, 0.95, f'r2 score: {performance:.2f}', transform=ax.transAxes, fontsize=14, verticalalignment='top')
        ax.set_xlabel('True values')
        ax.set_ylabel('Predicted values')
        ax.set_title('True vs predicted values of the target for the test set')

    return performance, fig_cm, cm


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
        # ToDo: fix error with LabelEncoder for multiclass classification
        # Do nothing for multiclass classification
        pass

    return df


# function for fitting standard scaler to the training data and performing PCA or spectral embedding
# returns the fitted scaler and the transformed training data
def reduce_dimensionality_train_test(train_data, test_data, target, ligand_numbers_column, substrate_names_column, transformer):
    # fit scaler to training data
    train_data = train_data.copy()
    train_data = train_data.reset_index(drop=True)
    X_train = train_data.drop([target, ligand_numbers_column, substrate_names_column], axis=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # fit transformer to the transformed training data
    X_train_transformed = transformer.fit_transform(X_train)
    # add original columns back to the transformed training data
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=[f'PC{i}' for i in range(1, X_train_transformed.shape[1] + 1)])
    X_train_transformed[[ligand_numbers_column, substrate_names_column, target]] = train_data[[ligand_numbers_column, substrate_names_column, target]]

    # transform test data
    test_data = test_data.copy()
    test_data = test_data.reset_index(drop=True)
    X_test = test_data.drop([target, ligand_numbers_column, substrate_names_column], axis=1)
    X_test = scaler.transform(X_test)
    X_test_transformed = transformer.transform(X_test)
    # add original columns back to the transformed test data
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=[f'PC{i}' for i in range(1, X_test_transformed.shape[1] + 1)])
    X_test_transformed[[ligand_numbers_column, substrate_names_column, target]] = test_data[[ligand_numbers_column, substrate_names_column, target]]
    return scaler, transformer, X_train_transformed, X_test_transformed
