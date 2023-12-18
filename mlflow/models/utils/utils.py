import mlflow
from mlflow.sklearn import log_model as sk_log_model



def set_experiment_run(
    tracking_uri,
    experiment_name,
    params:dict,
    metrics:dict,
    model,
    signature,
    artifacts:dict,
    tags:dict,
    input_example
    ):

    print("Setting experiment run...")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        
        mlflow.log_params(params)

        mlflow.log_metrics(metrics)

        if artifacts is not None:
            for artifact in artifacts:
                mlflow.log_artifact(artifact["local_path"], artifact["path"])

        mlflow.set_tags(tags)

        # Log the model
        model1_info = sk_log_model(
            sk_model=model,
            artifact_path=tags["artifact_path"],
            signature=signature,
            input_example=input_example,
            registered_model_name=tags["registered_model_name"],
        )

        print("Model info:")
        print(repr(model1_info))