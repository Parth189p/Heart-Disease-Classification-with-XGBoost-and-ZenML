# Heart Disease Classification with XGBoost and ZenML

![Project Status](https://img.shields.io/badge/Project%20Status-Active-green)

This project is an end-to-end architecture for heart disease classification using the XGBoost framework. The primary goal of this project is not only to build a high-quality model but also to demonstrate a scalable and production-ready machine learning project.

## Project Overview

The project focuses on utilizing ZenML for pipeline orchestration and ML ops tools to create a robust machine learning project. We have integrated MLflow with ZenML to streamline deployment, tracking, and more.

To address real-world use cases for predicting heart disease, a single model training is insufficient. Instead, I've developed an end-to-end pipeline for continuous prediction and deployment, complete with a data application that leverages the latest deployed model for business consumption.

The pipeline can be deployed to the cloud, scaled according to your needs, and guarantees that every pipeline run is tracked, from raw data input to features, results, machine learning model parameters, and prediction outputs. 

### Key Features
- ZenML for pipeline orchestration
- MLflow integration for tracking and deployment
- Continuous deployment workflow
- Model evaluation criteria
- Local MLflow deployment server

## Usage

### Training Pipeline

Our standard training pipeline consists of several steps:

1. **ingest_data:** This step ingests the data and creates a DataFrame.
2. **clean_data:** This step cleans the data and removes unwanted columns.
3. **train_model:** This step trains the model and saves it using MLflow autologging.
4. **evaluation:** This step evaluates the model and saves metrics using MLflow autologging.

### Deployment Pipeline

In addition to the training pipeline, I have a deployment pipeline (deployment_pipeline.py) for continuous deployment. It follows the same steps as the training pipeline and includes these additional ones:

1. **deployment_trigger:** This step checks whether the newly trained model meets deployment criteria.
2. **model_deployer:** This step deploys the model as a service using MLflow if deployment criteria are met.

ZenML's MLflow tracking integration logs hyperparameter values, the trained model, and evaluation metrics as MLflow experiment tracking artifacts into the local MLflow backend.

The MLflow deployment server runs locally as a daemon process. It continues to run in the background and updates to serve the latest model automatically when a new pipeline produces a model that meets the accuracy threshold.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Set up the necessary data sources.
4. Run the training and deployment pipelines as needed.

### You can run two pipelines 

- Training pipeline

```bash
python run_pipeline.py
```

- The continuous deployment pipeline

```bash
python run_deployment.py
```



