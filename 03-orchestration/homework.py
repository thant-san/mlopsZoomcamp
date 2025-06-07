import mlflow
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow.sklearn

# Load your data
df = pd.read_parquet('yellow_tripdata_2023-03.parquet')

# Preprocessing
df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
df = df[(df.duration >= 1) & (df.duration <= 60)]
df['PULocationID'] = df['PULocationID'].astype(str)
df['DOLocationID'] = df['DOLocationID'].astype(str)

# Prepare features and target
train_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')


dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
y_train = df['duration'].values

lr = LinearRegression()

# Set experiment (creates it if not exists)
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("yellow_trip_duration_experiment")

with mlflow.start_run():
    mlflow.set_tag("developer", "thant htoo san")
    mlflow.log_param("model_type", "LinearRegression")

    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mlflow.log_metric("rmse", rmse)

    # Save model and vectorizer to disk
    with open("model.pkl", "wb") as f_out:
        pickle.dump(lr, f_out)
    with open("dv.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)

    # Log artifacts
    mlflow.log_artifact("model.pkl")
    mlflow.log_artifact("dv.pkl")
    mlflow.sklearn.log_model(lr, artifact_path="model")