import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Prediction",layout="centered",initial_sidebar_state="collapsed")


@st.cache_resource
def load_model():
    with open("models/heart_model.bin","rb") as f_in:
        dv,model=pickle.load(f_in)
    return dv,model

dv,model=load_model()

st.title("❤️ Heart Disease Prediction (Dark Mode)")
st.write("Enter patient data below to estimate risk.")

with st.form("input_form"):

    st.header("Patient Details")

    age=st.number_input("Age",min_value=1,max_value=120,value=50)
    sex=st.selectbox("Sex",["M","F"])
    chestpaintype=st.selectbox("Chest Pain Type",["ATA-Atypical Angina", "NAP-Non-Anginal Pain", "ASY-Asymptomatic", "TA-Typical Angina"])
    restingbp=st.number_input("Resting BP",min_value=0,max_value=250,value=120)
    cholesterol=st.number_input("Cholesterol",min_value=0,max_value=700,value=200)
    fastingbs=st.selectbox("FastingBS (>120 mg/dl?)",[0,1])
    restingecg=st.selectbox("Resting ECG",["Normal", "ST-ST-T Wave Abnormality", "LVH-Left Ventricular Hypertrophy"])
    maxhr=st.number_input("Max Heart Rate",min_value=50,max_value=220,value=150)
    exerciseangina=st.selectbox("Exercise Angina",["Y","N"])
    oldpeak=st.number_input("Oldpeak",min_value=-3.0,max_value=10.0,value=1.0)
    st_slope=st.selectbox("ST Slope",["Up","Flat","Down"])

    submitted=st.form_submit_button("Predict Risk")

def predict_single(patient):
    X=dv.transform([patient])
    y_pred=model.predict(X)
    y_proba=model.predict_proba(X)[:,1]
    return int(y_pred[0]),float(y_proba[0])

def draw_risk_meter(prob):
    fig,ax=plt.subplots(figsize=(5,0.6))
    ax.barh([0],[prob],color=("red" if prob>0.7 else "orange" if prob>0.4 else "green"))
    ax.set_xlim(0,1)
    ax.set_yticks([])
    ax.set_xticks([0,0.25,0.5,0.75,1])
    ax.set_xlabel("Risk Level",color="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor('#0e1117')
    return fig

def draw_feature_importance():
    try:
        importances=model.feature_importances_
        names=dv.feature_names_
        sorted_idx=np.argsort(importances)

        fig,ax=plt.subplots(figsize=(6,4))
        ax.barh(np.array(names)[sorted_idx],importances[sorted_idx],color="#4CAF50")
        ax.set_title("Feature Importance",color="white")
        ax.tick_params(colors="white")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor('#0e1117')
        return fig
    except:
        return None

if submitted:
    patient_data={
        "age":age,
        "sex":sex,
        "chestpaintype":chestpaintype,
        "restingbp":restingbp,
        "cholesterol":cholesterol,
        "fastingbs":fastingbs,
        "restingecg":restingecg,
        "maxhr":maxhr,
        "exerciseangina":exerciseangina,
        "oldpeak":oldpeak,
        "st_slope":st_slope
    }

    pred,proba=predict_single(patient_data)

    st.subheader("📊 Probability Score")
    st.write(f"**Probability:** `{proba:.4f}`")

    st.pyplot(draw_risk_meter(proba))

    if pred==1:
        st.error("High Risk: The patient is likely to have heart disease.")
    else:
        st.success("Low Risk: The patient is unlikely to have heart disease.")

    st.write("---")
    st.subheader("Input Summary")
    st.json(patient_data)

    st.write("---")
    st.subheader("Feature Importance")

    fig_imp=draw_feature_importance()
    if fig_imp:
        st.pyplot(fig_imp)
    else:
        st.info("Feature importance not available for this model.")
