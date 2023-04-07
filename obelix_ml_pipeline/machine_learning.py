# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.express as px


# function to perform stratified shuffle split on substate_names_column, gridsearchcv and cross_val_score on model
# afterwards the best RF model and its performance is returned
def train_classifier(train_data, ligand_numbers_column, substrate_names_column, target, test_size, cv=5, scoring='balanced_accuracy', n_jobs=1, print_results=False):
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
    sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size, random_state=42)

    # create empty lists to store the performance of the models for each fold
    training_train_scores = []
    training_test_scores = []
    training_train_confusion_matrices = []
    training_test_confusion_matrices = []
    i = 1
    for train_index, test_index in sss.split(X, train_data[substrate_names_column]):
        # split the data into training and test set
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # create a gridsearchcv object
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, n_jobs=n_jobs, scoring=scoring)

        # fit the gridsearchcv object
        grid_search.fit(X_train, y_train)

        # get the best model
        best_model = grid_search.best_estimator_

        # calculate the performance of the best model on the training data
        training_train_balanced_accuracy = balanced_accuracy_score(y_train, best_model.predict(X_train))
        if print_results:
            print(f'Training performance on current fold {i}: {training_train_balanced_accuracy:.2f}')
        training_train_scores.append(training_train_balanced_accuracy)

        # calculate the performance of the best model on the test data
        training_test_balanced_accuracy = balanced_accuracy_score(y_test, best_model.predict(X_test))
        if print_results:
            print(f'Test performance on current fold {i}: {training_test_balanced_accuracy:.2f}')
        training_test_scores.append(training_test_balanced_accuracy)

        # calculate the confusion matrix of the best model on the training data
        training_train_confusion_matrix = confusion_matrix(y_train, best_model.predict(X_train))
        if print_results:
            print(f'Confusion matrix on current fold {i}: {training_train_confusion_matrix}')
        training_train_confusion_matrices.append(training_train_confusion_matrix)

        # calculate the confusion matrix of the best model on the test data
        training_test_confusion_matrix = confusion_matrix(y_test, best_model.predict(X_test))
        if print_results:
            print(f'Confusion matrix on current fold {i}: {training_test_confusion_matrix}')
        training_test_confusion_matrices.append(training_test_confusion_matrix)

        i += 1

    # calculate the mean and standard deviation of the performance across the folds
    training_train_scores_mean = np.mean(training_train_scores)
    training_train_scores_std = np.std(training_train_scores)
    training_test_scores_mean = np.mean(training_test_scores)
    training_test_scores_std = np.std(training_test_scores)

    # average the confusion matrices across the folds
    avg_training_train_confusion_matrix = np.mean(training_train_confusion_matrices, axis=0)
    avg_training_test_confusion_matrix = np.mean(training_test_confusion_matrices, axis=0)

    if print_results:
        print('Mean training performance: {:.2f} +/- {:.2f}'.format(training_train_scores_mean, training_train_scores_std))
        print('Mean test performance: {:.2f} +/- {:.2f}'.format(training_test_scores_mean, training_test_scores_std))

    # figures for the confusion matrices
    fig_cm, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.heatmap(avg_training_train_confusion_matrix, annot=True, fmt='f', ax=ax[0])
    ax[0].set_title('Training data')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    sns.heatmap(avg_training_test_confusion_matrix, annot=True, fmt='f', ax=ax[1])
    ax[1].set_title('Test data')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('Actual')

    # figure for the feature importance of the best model interactively with plotly sorted by importance
    feature_importances = pd.DataFrame(best_model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
    fig_fi = px.bar(feature_importances, x=feature_importances.index, y='importance', title='Feature importance of the best model')
    fig_fi.update_xaxes(title_text='Feature')
    fig_fi.update_yaxes(title_text='Importance')

    return best_model, training_train_scores_mean, training_train_scores_std, training_test_scores_mean, training_test_scores_std, avg_training_train_confusion_matrix, avg_training_test_confusion_matrix, fig_cm, fig_fi

    # # v2, new attempt
    # sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size, random_state=42)
    # train_indices, test_indices = next(sss.split(train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1), train_data[substrate_names_column]))
    # train_set = train_data.iloc[train_indices]
    # # train_set = train_data.drop(test_indices, axis=0)
    # test_set = train_data.iloc[test_indices]
    #
    # # Define features and target variable for training
    # X_train = train_set.drop([ligand_numbers_column, substrate_names_column, target], axis=1)
    # y_train = train_set[target]
    # param_grid = {
    #     'bootstrap': [False],
    #     'max_depth': [5, 50, 100],
    #     'max_features': [3, 5],
    #     'min_samples_leaf': [1, 2],
    #     'min_samples_split': [5, 10],
    #     'n_estimators': [50, 100]
    # }
    # # create a base random forest model
    # rf = RandomForestClassifier(random_state=42)
    # model = GridSearchCV(estimator=rf, param_grid=param_grid, cv=sss, n_jobs=n_jobs, scoring=scoring)
    # model.fit(X_train, y_train)
    #
    # # get best parameters
    # best_params = model.best_params_
    # best_score = model.best_score_
    # best_estimator = model.best_estimator_
    #
    # # print best parameters
    # if print_results:
    #     print('Best parameters: ', best_params)
    #     print('Best balanced accuracy score: ', best_score)
    # X_test = test_set.drop([ligand_numbers_column, substrate_names_column, target], axis=1)
    # y_test = test_set[target]
    # training_test_score = best_estimator.score(X_test, y_test)
    # if print_results:
    #     print('TRAINING TEST SET SCORE: ', training_test_score)
    #
    # # Get the cross-validation score on all folds
    # cv_scores = cross_val_score(best_estimator, train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1),
    #                             train_data[target], cv=sss)
    # print(f"CROSS-VALIDATION SCORES: {cv_scores}")
    # print(f"MEAN CROSS-VALIDATION SCORES: {cv_scores.mean()}")
    #
    # y_pred = best_estimator.predict(X_test)
    #
    # # print performance on test set
    # test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    # if print_results:
    #     print('Training test set performance: ', test_balanced_accuracy)
    #
    # # print confusion matrix
    # test_cm = confusion_matrix(y_test, y_pred)
    # if print_results:
    #     print('Training test set confusion matrix: ', test_cm)
    #
    # # print performance on training set
    # y_pred = model.predict(X_train)
    # y_true = y_train
    # train_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    # if print_results:
    #     print('Training train set performance: ', train_balanced_accuracy)
    #
    # # print confusion matrix
    # train_cm = confusion_matrix(y_true, y_pred)
    # if print_results:
    #     print('Training train set confusion matrix: ', train_cm)
    #
    #
    #
    # # # print cross validation scores
    # # scores = cross_val_score(model, train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1), train_data[target], cv=sss.split(train_data, groups=train_data[substrate_names_column], y=train_data[target]), scoring=scoring, n_jobs=n_jobs)
    # # print('Cross validation scores: ', scores)
    # # print('Mean cross validation score: ', scores.mean())
    #
    # #
    #
    # # plot confusion matrix
    # fig_cm, ax = plt.subplots(figsize=(10, 10))
    # sns.heatmap(test_cm, annot=True, fmt='d', linewidths=.5, ax=ax)
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    #
    # # plot feature importance
    # feature_importance = pd.DataFrame(
    #     {'feature': X_train.columns,
    #      'importance': model.best_estimator_.feature_importances_})
    # feature_importance = feature_importance.sort_values('importance', ascending=False)
    # fig_fi = px.bar(feature_importance, x='feature', y='importance', title='Feature importance')
    # return model, fig_cm, fig_fi, test_balanced_accuracy, train_balanced_accuracy, test_cm, train_cm

    # v1
    # # split the data into training and test set
    # sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size, random_state=42)
    # # feed stratified shuffle split to model
    # param_grid = {
    #     'bootstrap': [False],
    #     'max_depth': [5, 50, 100],
    #     'max_features': [3, 5],
    #     'min_samples_leaf': [1, 2],
    #     'min_samples_split': [5, 10],
    #     'n_estimators': [50, 100]
    # }
    # # create a base random forest model
    # # model = RandomForestClassifier(bootstrap=False, max_features=0.2, min_samples_leaf=1, min_samples_split=13, n_estimators=100, random_state=42)
    # rf = RandomForestClassifier(random_state=42)
    # model = GridSearchCV(estimator=rf, param_grid=param_grid, cv=sss.split(train_data, groups=train_data[substrate_names_column], y=train_data[target]), scoring=scoring, n_jobs=n_jobs)
    # model.fit(train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1), train_data[target])
    #
    # # get best parameters
    # best_params = model.best_params_
    #
    # # print best parameters
    # if print_results:
    #     print('Best parameters: ', best_params)
    #
    # # # print cross validation scores
    # # scores = cross_val_score(model, train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1), train_data[target], cv=sss.split(train_data, groups=train_data[substrate_names_column], y=train_data[target]), scoring=scoring, n_jobs=n_jobs)
    # # print('Cross validation scores: ', scores)
    # # print('Mean cross validation score: ', scores.mean())
    #
    # # print performance on test set
    # test_set = train_data.loc[list(sss.split(train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1), groups=train_data[substrate_names_column], y=train_data[target]))[-1][1]]
    # y_pred = model.predict(test_set.drop([ligand_numbers_column, substrate_names_column, target], axis=1))
    # y_true = test_set[target]
    # test_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    # if print_results:
    #     print('Test set performance: ', test_balanced_accuracy)
    #
    # # print confusion matrix
    # test_cm = confusion_matrix(y_true, y_pred)
    # if print_results:
    #     print('Test set confusion matrix: ', test_cm)
    #
    # # print performance on training set
    # y_pred = model.predict(train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1))
    # y_true = train_data[target]
    # train_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    # if print_results:
    #     print('Training set performance: ', train_balanced_accuracy)
    #
    # # print confusion matrix
    # train_cm = confusion_matrix(y_true, y_pred)
    # if print_results:
    #     print('Training set Confusion matrix: ', train_cm)
    #
    # # plot confusion matrix
    # fig_cm, ax = plt.subplots(figsize=(10, 10))
    # sns.heatmap(test_cm, annot=True, fmt='d', linewidths=.5, ax=ax)
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    #
    # # plot feature importance
    # feature_importance = pd.DataFrame({'feature': train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1).columns, 'importance': model.best_estimator_.feature_importances_})
    # feature_importance = feature_importance.sort_values('importance', ascending=False)
    # fig_fi = px.bar(feature_importance, x='feature', y='importance', title='Feature importance')
    # return model, fig_cm, fig_fi, test_balanced_accuracy, train_balanced_accuracy, test_cm, train_cm


# function to use trained classifier to predict on test data
def predict_classifier(test_data, ligand_numbers_column, substrate_names_column, target, model, print_results=False):
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

# # function to train a ML model
# def train_classifier(train_data, target, target_threshold, model, model_params, cv=5, scoring='balanced_accuracy', n_jobs=1):
#   # split the data into training and test set
#   sss = StratifiedShuffleSplit(n_splits=cv, test_size=0.2, random_state=0)
#   # feed stratified shuffle split to model
#   params = {'cv': sss, 'scoring': scoring, 'n_jobs': n_jobs}
#   model = GridSearchCV(model, model_params, cv=sss, scoring=scoring, n_jobs=n_jobs)
#   # get best parameters
#   best_params = model.best_params_
#   # train the model
#   model = model(**best_params)
#   model.fit(train_data.drop(target, axis=1), train_data[target])
#
#   # print best parameters
#     print(f'Best parameters: {model.best_params_}, performance {model.best_score_}')
#   # for train_index, test_index in sss.split(train_data, train_data[target]):
#   #   X_train, X_test = train_data.iloc[train_index], train_data.iloc[test_index]
#   #   y_train, y_test = train_data.iloc[train_index][target], train_data.iloc[test_index][target]
#   #   # perform grids search
#   # feed stratified shuffle split to model
#
# # perform grids search
#
#
#   # train the model
#   model = model(**model_params)
#   model.fit(X_train, y_train)
#
#   # predict the test set
#   y_pred = model.predict(X_test)
#
#   # calculate the performance metrics
#   accuracy = accuracy_score(y_test, y_pred)
#   balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
#   cm = confusion_matrix(y_test, y_pred)
#
#   # plot the confusion matrix
#   fig = px.imshow(cm)
#   fig.update_layout(    font=dict(
#           size=12,
#           color="black"),
#           xaxis_title='Predicted label',
#           yaxis_title='True label')
#   # px.plot(fig, filename='confusion_matrix.html')
#
#   # plot the ROC curve
#   # fig = px.imshow(cm)
#   # fig.update_layout(    font=dict(
#   #         size=12,
#   #         color="black"),
#   #         xaxis_title='Predicted label',
#   #         yaxis_title='True label')
#   # px.plot(fig, filename='confusion_matrix.html')
#
#   return model, accuracy, balanced_accuracy, fig
#
# # separate function to test the model
# def test_classifier(test_data, target, model):
#   # predict the test set
#   y_pred = model.predict(test_data)
#
#   # calculate the performance metrics
#   accuracy = accuracy_score(test_data[target], y_pred)
#   balanced_accuracy = balanced_accuracy_score(test_data[target], y_pred)
#   cm = confusion_matrix(test_data[target], y_pred)
#
#   # plot the confusion matrix
#   fig = px.imshow(cm)
#   fig.update_layout(    font=dict(
#           size=12,
#           color="black"),
#           xaxis_title='Predicted label',
#           yaxis_title='True label')
#   # px.plot(fig, filename='confusion_matrix.html')
#
#   return accuracy, balanced_accuracy, fig

def show_feature_importance(df, trained_model, target, substrate_names_column):
  df = df.drop([substrate_names_column, target], axis=1)
  # I assume that we only use RF or TPOT for now
  try:  # in the case of RF there is a property that contains feature importances
    importances = trained_model.feature_importances_
  except AttributeError:
    importances = trained_model.fitted_pipeline_.steps[-1][1].feature_importances_
  std = np.std([tree.feature_importances_ for tree in trained_model.estimators_], axis=0)
  forest_importances = pd.Series(importances, index=df.columns.values)

  fig = px.bar(forest_importances).update_traces(
      error_y={
          "type": "data",
          "symmetric": False,
          "array": std,
      }
  )
  fig.update_layout(    font=dict(
          size=12,
          color="black"),
          xaxis_title='Feature')
  # px.plot.bar(yerr=std, ax=ax)
  # ax.set_title("Feature importances Random Forest Regressor")
  # fig.tight_layout()
  fig.update_layout(showlegend=False)
  # fig.write_image(f"{path_to_google_drive_folder}/machine_learning/feature_importance_RF.png", scale = 5)

  fig.show()
  # positions = range(trained_model.feature_importances_.shape[0])
  # plt.bar(positions, trained_model.feature_importances_)
  # plt.show()