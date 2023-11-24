import logging
from abc import ABC,abstractmethod
import numpy as np
from typing import Union
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class DataStrategy(ABC):
    '''
    Abstract class defining strategy for handling data
    
    '''
    
    @abstractmethod
    def handle_data(self,data:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        pass
    
    
class Datapreprocessstrategy(DataStrategy):
    '''
    Strategy for preprocessing data
    '''
    
    def handle_data(self,data:pd.DataFrame)->pd.DataFrame:
        '''
        Preprocess data
        '''
        try:
            data = data.select_dtypes(include=[np.number])
            return data
        except Exception as e:
            logging.error("Error processing data:{}".format(e))
            raise e
        
                        
class DataDivideStrategy(DataStrategy):
    '''
    Strategy for deviding data into train and test
    ''' 
    def handle_data(self, data: pd.DataFrame) ->Union[pd.DataFrame,pd.Series]:
        '''
        divide data into train and test and apply standerd scaling
        '''
        try:
            x = data.iloc[:,0:13]
            y = data.iloc[:,13:14]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 10)
            # scaler = StandardScaler().fit(x_train_sc)
            # x_train = scaler.transform(x_train_sc)
            # x_test = scaler.transform(x_test_sc)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing the data:{}".format(e))
            raise e
        
class DataCleaning:
    '''
    Class for cleaning data which preprocess the data and divide it into train and test
    '''
    def __init__(self,data:pd.DataFrame, strategy : DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self)->Union[pd.DataFrame,pd.Series]:
        '''
        handle the data
        '''
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data:{}".format(e))
            raise e
        
        