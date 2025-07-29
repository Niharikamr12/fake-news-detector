import numpy as np, joblib
from src.preprocessing import clean_text

def predict_ensemble(texts):
    tfidf = joblib.load("models/tfidf.pkl")
    models = joblib.load("models/base_models.pkl")
    texts = [clean_text(t) for t in texts]
    X_vec = tfidf.transform(texts)
    preds = [m.predict_proba(X_vec)[:,1] for m in models]
    avg_preds = np.mean(preds, axis=0)
    return (avg_preds > 0.5).astype(int), avg_preds

