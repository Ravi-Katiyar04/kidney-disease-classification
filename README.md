# kidney-disease-classification-MLflow-DVC

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml
10. app.py

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/Ravi-Katiyar04/kidney-disease-classification-deep-learning-project.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
python -m venv myvenv
```

```bash
myvenv\Scripts\activate
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

- [MLflow tutorial](https://youtu.be/qdcHHrsXA48?si=bD5vDS60akNphkem)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/Ravi-Katiyar04/kidney-disease-classification.mlflow \
MLFLOW_TRACKING_USERNAME=Ravi-Katiyar04 \
MLFLOW_TRACKING_PASSWORD=40f60a5b1e6c3038bcb89f072b6ed6e1d691791c \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/Ravi-Katiyar04/kidney-disease-classification.mlflow

export MLFLOW_TRACKING_USERNAME=Ravi-Katiyar04 

export MLFLOW_TRACKING_PASSWORD=40f60a5b1e6c3038bcb89f072b6ed6e1d691791c

```

### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)

