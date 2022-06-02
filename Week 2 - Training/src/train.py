import argparse
import os
import pickle
from random import random
from xml.etree.ElementInclude import default_loader


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from yaml import parse
import mlflow



def load_pickle(filename:str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("week_2_random_forest")
    mlflow.sklearn.autolog()

    # max_depth = 10

    print(mlflow.get_tracking_uri())

    with mlflow.start_run():
        
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "valid.pkl"))

        

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
       
        # mlflow.log_param("max_depth", max_depth)
        # mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="../data/processed",
        help='the location where the processed NYC taxi trip data was saved.'
    )
    args = parser.parse_args()
    run(args.data_path)