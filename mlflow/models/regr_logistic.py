
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.utils import set_experiment_run


# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "newton-cg",
    "max_iter": 200,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)


signature = infer_signature(X_train, lr.predict(X_train))


tags = dict()
tags["artifact_path"] = "model1_for_iris"
tags["registered_model_name"] = "model1_for_iris"


set_experiment_run(
    tracking_uri="http://127.0.0.1:5000",
    experiment_name="experiment1",
    params=params,
    metrics={"accuracy": accuracy},
    model=lr,
    signature=signature,
    artifacts=None,
    tags=tags,
    input_example=X_train
    )







