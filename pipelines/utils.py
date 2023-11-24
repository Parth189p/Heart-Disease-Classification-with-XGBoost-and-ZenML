import logging
import pandas as pd
from src.data_cleaning import DataCleaning,Datapreprocessstrategy

def get_data_for_test():
    try:
        df = pd.read_csv(r"data/heart_desease_data.csv")
        df = df.sample(n = 10)
        preprocess_stretegy = Datapreprocessstrategy()
        data_cleaning = DataCleaning(df , preprocess_stretegy)
        df = data_cleaning.handle_data()
        result = df.to_json(orient ="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e