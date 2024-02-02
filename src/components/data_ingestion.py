import os 
import sys 
import pandas as pd 
from src.logger import logging
from src.exception import CustomException 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.data_transformation import DataTransformation 
# from src.components.data_transformation import DataTransformationConfig
from src.components.model_trianer import ModelTrainer 
@dataclass 
class DataIngestionConfig : 
    train_data_path  : str = os.path.join("artifacts" , "train.csv") 
    test_data_path  : str = os.path.join("artifacts" , "test.csv") 
    raw_data_path   : str = os.path.join("artifacts" , "data.csv") 

class DataIngestion: 
    def __init__(self) : 
        self.ingestion_path = DataIngestionConfig() 
    
    def intiate_data_ingestion(self): 
        logging.info( "Data Ingestion initiated ") 
        try : 
            df = pd.read_csv("data/stud.csv") 
            logging.info("Read Dataset as DataFrame ") 
            os.makedirs(os.path.dirname(self.ingestion_path.train_data_path ) , exist_ok= True ) 

            df.to_csv(self.ingestion_path.raw_data_path , index = False , header = True ) 

            logging.info( "Train Test Split Intiated ")
            train_set , test_set = train_test_split ( df , test_size= 0.2 , random_state= 42 )  

            train_set.to_csv (self.ingestion_path.train_data_path , index = False , header = True ) 
            test_set.to_csv(self.ingestion_path.test_data_path , index = False , header = True ) 


            logging.info("Ingestion of Data is Completed ") 

            return (
                self.ingestion_path.train_data_path , 
                self.ingestion_path.test_data_path
            )
        except Exception as e : 
            raise CustomException(e , sys ) 
            

if __name__ == "__main__" : 
    obj = DataIngestion() 
    train_data , test_data = obj.intiate_data_ingestion()

    data_transformation = DataTransformation() 
    train_arr , test_arr , _ = data_transformation.initiate_data_transformation(train_data , test_data )
    training_model = ModelTrainer()
    performance = training_model.initiate_model_training(train_arr , test_arr ) 
    print ( performance   ) 
