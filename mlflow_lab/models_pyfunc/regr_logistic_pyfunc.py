import sys
sys.path.append("..")
sys.path.append("../models")

import mlflow
import pandas as pd
from sklearn import datasets
from models.regr_logistic import model1_info, X_test, y_test


# Load the model back for predictions as a generic Python Function model
loaded_model1 = mlflow.pyfunc.load_model(model1_info.model_uri)

predictions = loaded_model1.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names
result = pd.DataFrame(X_test, columns=iris_feature_names)

result["actual_class"] = y_test
result["predicted_class"] = predictions

print(result[:4])