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
def train_classifier(train_data, ligand_numbers_column, substrate_names_column, target, test_size=0.2,cv=5, scoring='balanced_accuracy', n_jobs=1, print_results=False):
    # split the data into training and test set
    sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size, random_state=42)
    # feed stratified shuffle split to model
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
    model = GridSearchCV(estimator=rf, param_grid=param_grid, cv=sss.split(train_data, groups=train_data[substrate_names_column], y=train_data[target]), scoring=scoring, n_jobs=n_jobs)
    model.fit(train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1), train_data[target])

    # get best parameters
    best_params = model.best_params_

    # print best parameters
    if print_results:
        print('Best parameters: ', best_params)

    # # print cross validation scores
    # scores = cross_val_score(model, train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1), train_data[target], cv=sss.split(train_data, groups=train_data[substrate_names_column], y=train_data[target]), scoring=scoring, n_jobs=n_jobs)
    # print('Cross validation scores: ', scores)
    # print('Mean cross validation score: ', scores.mean())

    # print performance on test set
    test_set = train_data.loc[list(sss.split(train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1), groups=train_data[substrate_names_column], y=train_data[target]))[-1][1]]
    y_pred = model.predict(test_set.drop([ligand_numbers_column, substrate_names_column, target], axis=1))
    y_true = test_set[target]
    test_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    if print_results:
        print('Test set performance: ', test_balanced_accuracy)

    # print confusion matrix
    test_cm = confusion_matrix(y_true, y_pred)
    if print_results:
        print('Test set confusion matrix: ', test_cm)

    # print performance on training set
    y_pred = model.predict(train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1))
    y_true = train_data[target]
    train_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    if print_results:
        print('Training set performance: ', train_balanced_accuracy)

    # print confusion matrix
    train_cm = confusion_matrix(y_true, y_pred)
    if print_results:
        print('Training set Confusion matrix: ', train_cm)

    # plot confusion matrix
    fig_cm, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(test_cm, annot=True, fmt='d', linewidths=.5, ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # plot feature importance
    feature_importance = pd.DataFrame({'feature': train_data.drop([ligand_numbers_column, substrate_names_column, target], axis=1).columns, 'importance': model.best_estimator_.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    fig_fi = px.bar(feature_importance, x='feature', y='importance', title='Feature importance')
    return model, fig_cm, fig_fi, test_balanced_accuracy, train_balanced_accuracy, test_cm, train_cm


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
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])

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