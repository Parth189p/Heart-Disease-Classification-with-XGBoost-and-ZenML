from zenml import pipeline
from steps.ingest_data import Ingest_df
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=False)
def training_pipeline(data_path:str):
    df = Ingest_df(data_path)
    x_train,x_test,y_train,y_test = clean_data(df)
    model = train_model(x_train,x_test,y_train,y_test)
    rmse   = evaluate_model(model,x_test,y_test)
