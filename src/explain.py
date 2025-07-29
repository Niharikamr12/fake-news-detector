import lime
from lime.lime_text import LimeTextExplainer
import joblib
from src.preprocessing import clean_text

def explain_prediction(text):
    tfidf = joblib.load("models/tfidf.pkl")
    model = joblib.load("models/base_models.pkl")[0]  # Logistic Regression

    class_names = ['Real', 'Fake']
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba(texts):
        return model.predict_proba(tfidf.transform([clean_text(t) for t in texts]))

    exp = explainer.explain_instance(text, predict_proba, num_features=8)
    return exp.as_list()

