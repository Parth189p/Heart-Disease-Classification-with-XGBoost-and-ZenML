import logging
import pandas as pd
from zenml import step


class Ingestdata:
    '''
    ingesting data from data_path
    
    '''
    def __init__(self,data_path:str):
        self.data_path = data_path
        '''
        Args:
            data_path: path to data
            
        '''
    def get_data(self):
        '''
        ingesting data from data_path
        '''
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def Ingest_df(data_path:str) -> pd.DataFrame:
    '''
    ingesting data from data_path
    
    Args:
        data_path: path to data
    returns:
        pd.DataFrame: ingested data
        
    '''
    try:
        ingest_data = Ingestdata(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e