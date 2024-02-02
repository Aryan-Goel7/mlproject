from src.utils import load_object 

import pandas as pd 
import sys 
from src.exception import CustomException


class Prediction :
    def __init__(self , data : dict ) -> None : 
        self.data = data 
        print( data ) 
        self.df = pd.DataFrame()

    def dict_to_data_frame( self ) : 
        self.df = pd.DataFrame([self.data]) 
        print ( self.df ) 
        


    
    def preprocess(self) : 
        preprocessors = load_object("artifacts/preprocessors.pkl")
        self.dict_to_data_frame() 
        self.data = preprocessors.transform(self.df)
        print(self.data)
        return 

    def predict_value(self) : 
        try : 
            model = load_object("artifacts/model.pkl")
            self.preprocess()
            return model.predict(self.data)
        except Exception as e : 
            raise CustomException ( e , sys ) 
    


    
if __name__=="__main__" :
    data = {"gender": "male", "race_ethnicity": "group A", "parental_level_of_education": "some college", "lunch": "standard", "test_preparation_course": "completed", "reading_score": 75, "writing_score": 80}
    obj = Prediction(data) 
    print (obj.predict_value()[0]) 