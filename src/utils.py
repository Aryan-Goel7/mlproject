import os 
import sys 
import pandas as pd 
import numpy as np 
import dill 


from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV 
from src.exception import CustomException 


def save_object(file_path , obj ):
    try:
        dir_path = os.path.dirname(file_path) 
        os.makedirs(dir_path , exist_ok=True ) 


        with open(file_path,"wb") as file_obj:
            dill.dump(obj , file_obj )
    except Exception as e : 
        CustomException ( e , sys )  
def load_object(file_path) : 
    try : 
        with open(file_path,"rb") as file_obj :
            return dill.load(file_obj)

    except Exception as e : 
        raise CustomException(e , sys ) 
    

def evaluate_model( x_train , y_train , x_test , y_test , models , params  )  :
    
    try : 
        report = {} 
        # print ( models.items() ) 
        for model_name , model in models.items() : 
            # print ( model_name ) 

            grid_search = GridSearchCV( model , param_grid= params[model_name], n_jobs = -1 , scoring = "r2" )
            grid_search.fit (x_train , y_train )
            model.set_params(**grid_search.best_params_)
            model.fit(x_train , y_train) 
            y_train_pred = model.predict(x_train) 
            y_test_pred = model.predict(x_test) 
            train_model_score = r2_score( y_train , y_train_pred )
            test_model_score = r2_score ( y_test , y_test_pred ) 
            print ( model_name , [ test_model_score , train_model_score ] ) 
            
            report[model_name] =  [test_model_score , train_model_score ] 

        return report     
    except Exception as e :
        raise CustomException(e , sys ) 