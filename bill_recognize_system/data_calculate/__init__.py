"""
The :mod:`bill_recognize_system.data_calculate` module includes some
methods for predicting data
"""
from bill_recognize_system.data_calculate.regression_prediction import ModelName, RegressionPrediction, bill_predict

__all__: [
    "RegressionPrediction",
    "ModelName",
    "bill_predict"
]
