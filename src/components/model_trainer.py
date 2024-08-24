import os
import sys
from dataclasses import dataclass
import numpy as np

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj, evaluate_models

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            # logging.info(f"train_array: {train_array}")
            # logging.info(f"test_array: {test_array}")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost Classfier": CatBoostRegressor(verbose = False),
                "Adaboost Classifier":AdaBoostRegressor() 
            }


            model_report:dict = evaluate_models(
                X_train = X_train, 
                y_train = y_train,
                X_test = X_test,
                y_test =  y_test,
                models = models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best Model found")
            logging.info("Best model found")

            save_obj(
                file_path=self.model_trainer_config.train_model_file_path, 
                obj= best_model
            )

            predicted = best_model.predict(X_test)
            r2_sc = r2_score(y_test, predicted)

            return r2_sc

        except Exception as e:
            raise CustomException(e, sys)


