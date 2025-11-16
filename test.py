import requests

url='https://heartdisease-pred.onrender.com/predict'
patient = {
    "age": 46,
    "sex": "M",
    "chestpaintype": "ASY",
    "restingbp": 118,
    "cholesterol": 186,
    "fastingbs": 0,
    "restingecg": "Normal",
    "maxhr": 124,
    "exerciseangina": "N",
    "oldpeak": 0.0,
    "st_slope": "Flat"
}
# 46,M,ASY,118,186,0,Normal,124,N,0,Flat,1
response = requests.post(url, json=patient)
pred = response.json()

print(pred)

if pred["heartdisease"]:
    print("Patient likely has heart disease. Recommend cardiology follow-up.")
else:
    print("Patient unlikely to have heart disease.")
