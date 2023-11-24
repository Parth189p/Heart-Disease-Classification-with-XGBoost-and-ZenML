import logging
import pandas as pd
from zenml import step
from src.model_dev import XGBoostModel
from .config import ModelNameConfig
from sklearn.base import ClassifierMixin
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
# @step
def train_model(
    x_train:pd.DataFrame,
    x_test:pd.DataFrame,
    y_train:pd.DataFrame,
    y_test:pd.DataFrame,
    config:ModelNameConfig
) -> ClassifierMixin:
    '''
    Train the model on the given dataset.
    
    Args:
        x_train:pd.DataFrame,
        x_test:pd.DataFrame,
        y_train:pd.DataFrame,   
        y_test:pd.DataFrame
        
    '''
    model  = None
    try:
        if config.model_name == 'XGboost':
            mlflow.xgboost.autolog()
            model = XGBoostModel()
            train_model = model.train(x_train, y_train)
            return train_model
        else:
            raise ValueError('Model {} not supported'.format(config.model_name))
    except Exception as e:
        logging.error('Error in training model: {}'.format(e))
        raise e
    