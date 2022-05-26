# imports
from cmath import exp
# from sklearn import set_config; set_config(display='diagram')
from sklearn.model_selection import train_test_split

from TaxiFareModel.data import get_data, clean_data, df_optimized
from TaxiFareModel.encoders import DistanceTransformer,TimeFeaturesEncoder

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

import mlflow
from  mlflow.tracking import MlflowClient
from memoized_property import memoized_property

from google.cloud import storage
import joblib

import numpy as np



class Trainer():
    def __init__(self, X, y, experiment_name, BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, MODEL_NAME, MODEL_VERSION, STORAGE_LOCATION):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = experiment_name
        self.BUCKET_NAME =  BUCKET_NAME
        self.BUCKET_TRAIN_DATA_PATH = BUCKET_TRAIN_DATA_PATH
        self.MODEL_NAME = MODEL_NAME
        self.MODEL_VERSION = MODEL_VERSION
        self.STORAGE_LOCATION = STORAGE_LOCATION


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # transformer and scaler
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
            ])

        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ])

        #preproc pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")

        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
            ])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return np.sqrt(((y_pred - y_test)**2).mean())

    # cloud storage

    def upload_model_to_gcp(self):
        client = storage.Client()
        bucket = client.bucket(self.BUCKET_NAME)
        blob = bucket.blob(self.STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')

    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        reg = self.run()
        joblib.dump(reg, 'model.joblib')
        print("saved model.joblib locally")

        # Implement here
        self.upload_model_to_gcp()
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


    # memoized properties

    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = mlflow.set_tracking_uri("https://mlflow.lewagon.ai/")
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)










if __name__ == "__main__":

    ### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

    BUCKET_NAME = 'wagon-data-871-laia'
    ##### Data  - - - - - - - - - - - - - - - - - - - - - - - -
    # train data file location
    # /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
    # or if you want to use the full dataset (you need need to upload it first of course)
    BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'
    # model folder name (will contain the folders for all trained model versions)
    MODEL_NAME = 'taxifare'
    # model version folder name (where the trained model.joblib file will be stored)
    MODEL_VERSION = 'v1'
    STORAGE_LOCATION = 'models/TaxiFareModel/model.joblib'


    # get data
    print('getting data')
    df = get_data()

    # clean data
    print('cleaning data')
    df = clean_data(df)

    print('optimizing')
    df = df_optimized(df)

    # set X and y
    print('setting x and y')
    target = 'fare_amount'

    X = df.drop(target, axis=1)
    y = df[target]

    # hold out
    print('holdout')
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1)

    # train
    print('training')
    exp_name = '[DE][BE][Laiagdla]Taxi0.1'
    trainer = Trainer(X_train, y_train, exp_name, BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, MODEL_NAME, MODEL_VERSION, STORAGE_LOCATION)
    print('\n saving model CGP')
    trainer.save_model()


    # evaluate

    mse = trainer.evaluate(X_test, y_test)
    print(f'score: {mse}')
