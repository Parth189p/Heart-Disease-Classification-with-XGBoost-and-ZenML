import pandas as pd
import numpy as np
import json
from zenml import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.ingest_data import Ingest_df
from steps.model_train import train_model
from .utils import get_data_for_test
from materializer.custom_materializer import cs_materializer

# from zenml.integrations.mlflow.steps import MLFlowDeployerConfig
# from zenml.integrations.mlflow.steps import mlflow_deployer_step
# from zenml.steps import BaseStepConfig

docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    """
    Deployment trigger configuration
    """
    min_accuracy : float = 0.40


@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data
    
@step
def deployment_trigger(
    accuracy : float ,
    config: DeploymentTriggerConfig
) :
    """
    Implements the simple model deployment trigger that looks at the input model accuracy and decide it is good enough to deploy or not
    """
    return accuracy > config.min_accuracy


class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True
    
# model_deployer = mlflow_deployer_step(name="model_deployer")
    

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    # data.pop("columns")
    # data.pop("index")

    df = pd.DataFrame(data["data"])
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    # data.pop("columns")
    # data.pop("index")
    df = pd.DataFrame(data["data"])
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


@pipeline(enable_cache = False , settings={'docker': docker_settings})
def continuous_deployment_pipeline(
    data_path : str,
    min_accuracy : float = 0.40,
    workers : int = 1 ,
    timeout : int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    # model_deployer=model_deployer(config=MLFlowDeployerConfig(workers=3))
    df = Ingest_df(data_path = data_path)
    x_train,x_test,y_train,y_test = clean_data(df)
    model = train_model(x_train,x_test,y_train,y_test)
    rmse   = evaluate_model(model,x_test,y_test)
    deployment_decision = deployment_trigger(rmse)
    mlflow_model_deployer_step(
        model = model,
        deploy_decision=deployment_decision,
        workers = workers,
        timeout = timeout, 
    )
    

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)