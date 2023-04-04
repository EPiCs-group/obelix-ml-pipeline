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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.express as px


def split_and_train(split_object, ml_model, train_data, substrate_names_column, target):
  split = split_object
  for train_index, val_index in split.split(train_data, train_data[substrate_names_column]):
      train_data_split = train_data.iloc[train_index]
      val_data_split = train_data.iloc[val_index]
  features_data = train_data_split.drop([substrate_names_column, target], axis=1)
  target_data = train_data_split[target]

  trained_model = ml_model.fit(features_data, target_data)

  return trained_model

def prepare_binary_classification_data(df, target, target_threshold):
  # for classification, transform the data based on
  df.loc[df[target] < target_threshold, target] = 0
  df.loc[df[target] > target_threshold, target] = 1
  df[target] = df[target].astype(np.int64)

  return df


def train_and_validate_classifier(train_data, target, target_threshold, list_of_substrates, test_size, ml_model, substrate_names_column):
  # conversion of target property based on threshold should be done outside of this function already
  # Create Stratified Shuffle Split object
  print('Training classifier')
  split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
  trained_model = split_and_train(split, ml_model, train_data, substrate_names_column, target)
  return trained_model

def predict_and_evaluate_performance(test_data, trained_model, list_of_substrates, substrate_names_column, target):
  # for prediction on unseen substrate
  print('Testing classifier')
  prediction_features = test_data.drop([substrate_names_column, target], axis=1)
  predictions = trained_model.predict(prediction_features)
  y_true, y_pred = test_data[target].astype(int), predictions
  # Assess the accuracy of predictions
  accuracy = accuracy_score(y_true, y_pred)
  balanced_accuracy = balanced_accuracy_score(y_true, y_pred)  # takes imbalanced dataset into account
  print("Balanced accuracy", balanced_accuracy)
  print("Accuracy:", accuracy)
  score = r2_score(y_true, y_pred)
  print('R2 score:', score)
  confusion_mat = confusion_matrix(y_true, y_pred)
  print('Confusion matrix', confusion_mat)

def train_and_validate_regressor(train_data, target, list_of_substrates, test_size, ml_model, substrate_names_column):
  # Create Stratified Shuffle Split object
  print('Training regressor')
  split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
  trained_model = split_and_train(split, ml_model, train_data, substrate_names_column, target)
  return trained_model

def predict_and_evaluate_performance_regressor(test_data, trained_model, list_of_substrates, substrate_names_column, target, title=None, ax=None):
  print('Testing regressor')
  predictions = trained_model.predict(test_data.drop([substrate_names_column, target], axis=1))
  y_true, y_pred = test_data[target], predictions
  score = r2_score(y_true, y_pred)
  print('R2 score:', score)
  print('Plotting results')
  if ax is None:
    fig, ax = plt.subplots()
  sns.regplot(x=y_true, y=y_pred, ax=ax)
  ax.set_xlabel('True')
  ax.set_ylabel('Predicted')
  ax.set_title(title)
  ax.text(0.05, 0.9, f'$R^2$={r2_score(y_true, y_pred):.2f}', transform=ax.transAxes)
  fig.show()


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