
import os 
import sys 
from dataclasses import dataclass 


from catboost import CatBoostRegressor 
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor  


from src.exception import CustomException 
from src.logger import logging 
from src.utils import save_object, evaluate_model 




@dataclass 
class ModelTrainerConfig : 
    model_trainer_file_path = os.path.join("artifacts","model.pkl") 


class ModelTrainer : 
    def __init__(self) -> None:
        self.model_trainer_path = ModelTrainerConfig()
    
    def initiate_model_training(self, train_array , test_array  ) : 
        try : 

            logging.info( "Splitting data in test and train array ") 
            X_train , y_train , X_test , y_test = (
                train_array[ : , :-1] ,
                train_array [ : , -1 ],
                test_array [ : , :-1] ,
                test_array [ : , -1 ]
            )

            models = {
                'CatBoost': CatBoostRegressor(silent = True ),
                'AdaBoost': AdaBoostRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'DecisionTree': DecisionTreeRegressor(),
                'XGBoost': XGBRegressor(),
                'KNeighbors': KNeighborsRegressor(),
                'RandomForest': RandomForestRegressor()           
            }


            hyper_parameters= {
                'CatBoost': {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [6, 8, 10],
                    'iterations': [100, 200, 300]
                },
                'AdaBoost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                },
                'GradientBoosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'RandomForest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'KNeighbors': {
                    'n_neighbors': [5, 10, 15],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                'DecisionTree': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'XGBoost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            }
            models_performance = evaluate_model(x_train = X_train , y_train = y_train , x_test = X_test , y_test = y_test , models = models , params = hyper_parameters ) 
            print ( models_performance )
            models_ranking = sorted ( models_performance.items() , key = lambda key : key[1][0] , reverse = True ) 
            
            # print ( models_ranking[0][1][0]  ) 
            if ( models_ranking[0][1][0] < 0.6 ) : 
                raise CustomException (" No model found performing well on the dataset ")
            else :
                best_model_name = models_ranking[0][0]
                print(best_model_name)
                best_model = models[best_model_name] 

                save_object (
                    file_path = self.model_trainer_path.model_trainer_file_path, 
                    obj = best_model
                )

                predicted = best_model.predict( X_test ) 

                score = r2_score (predicted , y_test ) 
                return score 
            
        except Exception as e: 
            CustomException ( e , sys ) ;  

         