from datetime import timedelta
from sched import scheduler
from numpy import timedelta64
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.metrics import mean_squared_error

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import xgboost as xgb

import mlflow

from prefect import flow, task
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners.subprocess import SubprocessFlowRunner
from datetime import timedelta



def read_dataframe(filename, pickup_time, drop_off_time, pu_id, do_id):
    df = pd.read_parquet(filename)
    df['duration'] = pd.to_datetime(df[drop_off_time]) - pd.to_datetime(df[pickup_time])  # calculate duration and convert to minutes
    df['duration'] = df['duration'].apply(lambda td: td.total_seconds()/60)
 
    df = df[((df.duration>=1) & (df.duration<=60))]       # filter to only duration within and less than hour

    categorical = [pu_id, do_id]

    df[categorical] = df[categorical].fillna(-1)
    df[categorical] = df[categorical].astype(str)
    return df

@task
def add_features(train_path,
                 test_path):


    # # Specify the location of the file
    # green_trip_data_Jan_2021 = '../data/green_tripdata_2021-01.parquet'
    # green_trip_data_Feb_2021 = '../data/green_tripdata_2021-02.parquet'

    # fix issue with path 
    # train_path = os.path.normpath(os.path.join(os.path.dirname(__file__), train_path))
    # test_path = os.path.normpath(os.path.join(os.path.dirname(__file__), test_path))

    print(__file__)
    print(os.getcwd())
    # /Users/dsn/Downloads/MLOps_Zoomcamp/Week 3 - Orchestration/data/green_tripdata_2021-01.parquet

    df_train = read_dataframe(train_path,
                                pickup_time="lpep_pickup_datetime",
                                drop_off_time='lpep_dropoff_datetime',
                                pu_id ='PULocationID',
                                do_id='DOLocationID'
                                )
    df_test = read_dataframe(test_path,
                                pickup_time="lpep_pickup_datetime",
                                drop_off_time='lpep_dropoff_datetime',
                                pu_id ='PULocationID',
                                do_id='DOLocationID'
                                )
    print(len(df_train), len(df_test))

    df_test['PU_DO'] = df_test['PULocationID'] + '_' + df_test['DOLocationID']
    df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']

    

    numerical = ['trip_distance']
    categorical= ['PU_DO'] # ['PULocationID', 'DOLocationID',

    dict_train = df_train[numerical + categorical].to_dict(orient='records')
    dict_eval = df_test[numerical + categorical].to_dict(orient='records')


    dv = DictVectorizer()
    X_train = dv.fit_transform(dict_train) #this returns a sparse cmr matrix
    X_eval = dv.transform(dict_eval)

    Y_train = df_train['duration'].values
    Y_eval = df_test['duration'].values

    # print(X_train)
    return X_train, X_eval, Y_train, Y_eval, dv

# print()
##############
# Modelling Section 
 
@task
def train_model_search(train, valid, y_val):

    def objective(params):

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round =500,
            evals =[(valid ,"validation")],
            early_stopping_rounds =50
        )

        y_pred = booster.predict(valid)
        rmse=mean_squared_error(y_val, y_pred, squared=False)
        # mlflow.log_metric("rmse", rmse)

        return {'loss': rmse,'status':STATUS_OK}

    search_space ={

        "max_depth": scope.int(hp.quniform('max_depth',4,100,1)),
        "learning_rate": hp.loguniform('learning_rate', -3, 0),
        "reg_alpha": hp.loguniform("reg_alpha", -5,-1),
        "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        "min_child_weight": hp.loguniform('min_child_weight', -1, 3),
        "objective": 'reg:linear',
        'seed': 42
    }

    best_result  = fmin(
        fn=objective,
        space=search_space,
        algo = tpe.suggest,
        max_evals=1,
        trials=Trials()
    )
    return

@task
def train_best_model(train, valid, y_eval): 
    best_params ={
        'learning_rate':	0.050492944008818104,
            'max_depth'	:4,
            'min_child_weight'	:2.287940332571657,
            'objective':	'reg:linear',
            'reg_alpha':	0.03134130292019771,
            'reg_lambda':	0.006582082224902776,
            'seed':	42
    }

    booster = xgb.train(
                params=best_params,
                dtrain=train,
                num_boost_round =100,
                evals =[(valid ,"validation")],
                early_stopping_rounds =50
    )
    
    y_pred = booster.predict(valid)
    rmse = mean_squared_error(y_eval, y_pred, squared=False)
    # mlflow.log_metric("rmse", rmse)

    # with open("../models/preprocessor.b", "wb") as f_out:
    #     pickle.dump(dv, f_out)

@flow
def main(train_path = './data/green_tripdata_2021-01.parquet',
         test_path = './data/green_tripdata_2021-02.parquet'):

    X_train, X_val , y_train, y_eval, dv =  add_features(train_path, test_path).result()
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_eval)
    train_model_search(train, valid, y_eval)
    train_best_model(train, valid, y_eval)



main()
# print(os.path.join(os.path.dirname(__file__),'../data/green_tripdata_2021-01.parquet'))
# print(os.path.normpath(os.path.join(os.path.dirname(__file__),'../data/green_tripdata_2021-01.parquet')))



# DeploymentSpec(

#     flow= main,
#     name="model_training",
#     schedule =IntervalSchedule(interval=timedelta(minutes=5)),
#     tags=['ml']
#     )
