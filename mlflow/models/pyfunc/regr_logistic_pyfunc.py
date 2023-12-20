import sys
sys.path.append("..")
sys.path.append("../models")


import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

#Get-Childitem env: 
#$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

#logged_model = 'runs:/7c6f53ccf5a343acb57447652e0d913e/model1_for_iris'
logged_model = 'models:/model1_for_iris/3'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
res = loaded_model.predict(pd.DataFrame([[
      6.7,
      3.3,
      5.7,
      2.5
    ]]))
  
print(res)


"""
5.7,
      3,
      4.2,
      1.2
      
out=1


6.7,
      3.3,
      5.7,
      2.5
  
out=2
"""


"""

predictions = loaded_model1.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names
result = pd.DataFrame(X_test, columns=iris_feature_names)

result["actual_class"] = y_test
result["predicted_class"] = predictions

print(result[:4])

print(res)
"""