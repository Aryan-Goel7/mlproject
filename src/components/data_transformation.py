from dataclasses import dataclass 

import sys 
import os 
import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder,StandardScaler 

from src.exception import CustomException 
from src.logger import logging 
from src.utils import save_object 


@dataclass 
class DataTransformationConfig : 
    # DataIngestionObj = DataIngestion() 
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessors.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() 
    
    def get_data_transformer_object(self):
        try: 
            num_columns = [ 'reading_score', 'writing_score']

            cat_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline( 
                steps = [
                    ("Imputer" , SimpleImputer(strategy="median") ),
                    ("Scaler",StandardScaler(with_mean=False) )
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer" , SimpleImputer(strategy="most_frequent")),
                    ("one_hot_scaler",OneHotEncoder()),
                    ("standard_scaler", StandardScaler(with_mean= False ))
                ]
            )
            logging.info(" Numerical Columns : {} ".format(num_columns ))
            logging.info("Categorical Columns : {} ".format(cat_columns) ) 


            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline" , num_pipeline , num_columns ), 
                    ("cat_pipeline" , cat_pipeline , cat_columns )
                ]
            )
            logging.info("Preproccesor Formed") 
            return preprocessor
        except Exception as e : 
            raise CustomException(e , sys ) 
    
    def initiate_data_transformation(self , train_data_path , test_data_path ) :

        try:
            train_df=pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path) 


            logging.info("Train and Test Data are Loaded Successfully ") 

            logging.info("Creating a preprocessing Object ") 

            preprocessing_obj = self.get_data_transformer_object() 

            # logging.info( "train_df columns : {}  test_df columns : {} ".format(train_df.columns , test_df.columns)) 

            target_column_name="math_score"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            ) 

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_data_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_data_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path , 
                obj = preprocessing_obj 
            )
            logging.info("Modelled is saved")

            return(
                train_data_arr ,
                test_data_arr , 
                self.data_transformation_config.preprocessor_obj_file_path
            )





        except Exception as e : 
            raise CustomException(e , sys )


 
