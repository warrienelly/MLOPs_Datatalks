import pickle
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners.subprocess import SubprocessFlowRunner

from datetime import date as dt



@task
def read_data(path):
    df = pd.read_parquet(path)
    return df
@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
        # print(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
        # print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df
@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date):
    # get date month
    logger = get_run_logger()
    if date is None:
        date_dt = dt.today
        train_date = date_dt.replace(month=date_dt.month-2)
        test_date =  date_dt.replace(month=date_dt.month-1)
    else:    
        date_dt = dt.fromisoformat(date)
        train_date = date_dt.replace(month=date_dt.month-2)
        test_date =  date_dt.replace(month=date_dt.month-1)

    train_path = f"../data/fhv_tripdata_{train_date.year}-{str(train_date.month).zfill(2)}.parquet"
    test_path =   f"../data/fhv_tripdata_{test_date.year}-{str(test_date.month).zfill(2)}.parquet"
    
    logger.info(f'Train path is : {train_path}')
    logger.info(f'Test path is : {test_path}')
    return train_path, test_path
    


@flow
def main(date: str):

    categorical = ['PUlocationID', 'DOlocationID']

    train_path , val_path = get_paths(date).result()


    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    with open(f'../models/model-{date}.bin', 'wb') as f_out:
        pickle.dump(lr, f_out)
    with open(f'../models/dv-{date}.bin', 'wb') as dv_out:
        pickle.dump(dv, dv_out)

# main(date="2021-08-15")
DeploymentSpec(
    flow=main,
    name="model_NYC_duration",
    schedule= CronSchedule(
        cron = "0 9 15 * *"),
    flow_runner = SubprocessFlowRunner(),
    # tag=['ml_NYC']
)

