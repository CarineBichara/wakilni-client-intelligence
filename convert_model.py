import joblib
import sklearn

print("Streamlit sklearn version:", sklearn.__version__)

model = joblib.load("churn_best_model.pkl")
joblib.dump(model, "churn_best_model_streamlit.pkl")

print("Model converted successfully!")
