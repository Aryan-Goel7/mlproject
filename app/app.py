from fastapi import FastAPI
from src.pipelines.predict_pipeline import Prediction
app = FastAPI()



@app.get("/", tags = ["ROOT"])
async def root() -> str : 
    return "Server is Active" 


@app.post("/predict" , tags = ["Data"]) 
def predict(data : dict ):
    obj = Prediction(data)
    predicted_value = obj.predict_value()[0] 
    return predicted_value
    


