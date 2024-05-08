# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

from dataclasses import dataclass

import pandas as pd


# dataclass for the results of the prediction
@dataclass
class PredictionResults:
    best_model: object
    training_best_model_performance: float
    training_test_scores_mean: float
    training_test_scores_std: float
    target_threshold: float
    fig_cm: object
    fig_fi: object
    df_fi: pd.DataFrame
    testing_performance_test: float
    testing_confusion_fig: object
    testing_cm_test: object
    # the following are for objective 4, in which a randomly (shrinking) subset of the data is used
    train_data: object = None
    test_data: object = None
    random_seed: int = None
    randomly_chosen_fraction: float = None
