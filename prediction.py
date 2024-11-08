
import joblib

def predict(data):
    reg = joblib.load('rf_model.joblib')
    return reg.predict(data)