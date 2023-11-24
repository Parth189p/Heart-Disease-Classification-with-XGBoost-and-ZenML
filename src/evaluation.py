import logging
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from abc import ABC , abstractmethod

class Evaluation(ABC):
    '''
    Abstract Class defining the strategy for evaluating model performance
    
    '''
    
    @abstractmethod
    def calculate_score(self,y_true: np.ndarray,y_pred: np.ndarray):
        '''
        Calculate the score for the model
        Args:
            y_true: True labels
            y_pred: predicted labels
        Returns:
            None
        
        '''
        pass
    
    
class MSE(Evaluation):
    '''
    Evaluation strategy that uses Mean Squared Error (MSE)    
    '''
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        try:
            logging.info('Calculating MSE')
            mse = mean_squared_error(y_true, y_pred)
            logging.info('MSE: {}".'.format(mse))
            return mse
        except Exception as e:
            logging.error('Error calculating MSE: {}'.format(e))
            raise e
        
class R2(Evaluation):
    '''
    Evaluation strategy that uses R2 Score
    
    '''
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            r2_score: float
        """
        try:
            logging.info('Calculating R2 score')
            r2 = r2_score(y_true, y_pred)
            logging.info('R2 score:{}'.format(r2))
            return r2
        except Exception as e:
            logging.error('Error calculating R2 score: {}'.format(e))
            raise e
        
        
class RMSE(Evaluation):
    '''
    Evaluation strategy that uses Root Mean Squared Error (RMSE)
    
    '''
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            rmse: float
        """
        try:
            logging.info('Calculating RMSE')
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info('RMSE: {}'.format(rmse))
            return rmse
        except Exception as e:
            logging.error('Error calculating RMSE: {}'.format(e))
            raise e
        
        
class Classification_Report(Evaluation):
    '''
    Evaluation strategy that uses classification report
    
    '''
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            classification_results
        """
        try:
            logging.info('Calculating classification_results')
            cr =classification_report(y_true, y_pred)
            logging.info('Classification_report:{}'.format(cr))
            return cr
        except Exception as e:
            logging.error('Error calculating classification report: {}'.format(e))
            raise e
        
class Confusion_Metrix(Evaluation):
    '''
    Evaluation strategy that uses confusion metrix
    
    '''
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            confusion_score
        """
        try:
            logging.info('Calculating confusion metrix')
            cm =confusion_matrix(y_true, y_pred)
            logging.info('COnfusion Metrix:{}'.format(cm))
            return cm
        except Exception as e:
            logging.error('Error calculating confusion metrix: {}'.format(e))
            raise e