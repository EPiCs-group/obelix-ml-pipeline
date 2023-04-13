# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

from dataclasses import dataclass


# dataclass for the results of the prediction
@dataclass
class PredictionResults:
    best_model: object
    training_best_model_performance: float
    training_test_scores_mean: float
    training_test_scores_std: float
    fig_cm: object
    fig_fi: object
    testing_performance_test: float
    testing_confusion_fig: object
    testing_cm_test: object
