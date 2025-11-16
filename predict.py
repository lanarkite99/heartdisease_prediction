import pickle
from fastapi import FastAPI
from pydantic import BaseModel,Field
import uvicorn
import numpy as np

app=FastAPI( title='Heart Disease Prediction')

@app.get("/")
def home():
    return {"status": "ok", "message": "Heart Disease Prediction API is live"}


class patient(BaseModel):
    age:int=Field(...,ge=0,le=130)
    sex:str=Field(...,min_length=1,max_length=1)
    chestpaintype:str=Field(...,min_length=1,max_length=3)
    restingbp:float
    cholesterol:int=Field(...,ge=0,le=600)
    fastingbs:int
    restingecg:str
    maxhr:float
    exerciseangina:str
    oldpeak:float
    st_slope:str

class PredictionResponse(BaseModel):
    heartdisease:int
    probability:float



with open ('models/heart_model.bin','rb') as f_in:
    dv,model=pickle.load(f_in)

def predict_single(patient:dict):
    X=dv.transform([patient])
    y_pred=model.predict(X)
    y_proba=model.predict_proba(X)
    pred = int(np.ravel(y_pred)[0])
    proba = float(np.ravel(y_proba[:, 1])[0])
    return pred, proba

@app.post('/predict',response_model=PredictionResponse)
def predict(patient:patient):
    pred,proba=predict_single(patient.model_dump())
    return PredictionResponse(heartdisease=pred,probability=proba)

if __name__=='__main__':
    uvicorn.run(app,host='0.0.0.0',port=8000)

