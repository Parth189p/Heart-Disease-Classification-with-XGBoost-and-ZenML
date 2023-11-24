import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataDivideStrategy,Datapreprocessstrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df : pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"x_train"],
    Annotated[pd.DataFrame,"x_test"],
    Annotated[pd.DataFrame,"y_train"],
    Annotated[pd.DataFrame,"y_test"],
]:
    '''
    Cleans data and divide intp train and test
    
    Args:   
        df : raw data
    returns:
        x_train : Training data
        x_test : Testing data
        y_train : Training labels
        y_test : Testing labels
    '''
    
    try:
        preprocess_strategy = Datapreprocessstrategy()
        data_cleaning = DataCleaning(df,preprocess_strategy)
        process_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(process_data,divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        return x_train,x_test,y_train,y_test
    except Exception as e:
        logging.error("Error cleaning dara:{}".format(e))
        raise e
    
        
        
        
        
        