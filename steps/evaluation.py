import logging
import pandas as pd
from sklearn.base import ClassifierMixin
from zenml import step
from typing_extensions import Annotated
from typing import Tuple
import numpy as np
import mlflow
from zenml.client import Client
from src.evaluation import RMSE

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)

def evaluate_model(model :ClassifierMixin,
                x_test:pd.DataFrame,
                y_test:pd.DataFrame,
    ) -> np.float64:
    '''
    Evaluate the model on ingested data
    Args:
        df: The ingested data
    
    '''
    try:
        predictions = model.predict(x_test)
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, predictions)
        mlflow.log_metric("rmse", rmse)
        
        # cm_class = Confusion_Metrix()
        # cm = cm_class.calculate_score(y_test, predictions)
        
        # cr_class = Classification_Report
        # cr = cr_class.calculate_score(y_test, predictions)
        
        return rmse
    except Exception as e:
        logging.error("Error in Evaluating the model:{}".format())
        raise e