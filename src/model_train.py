import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib, os
from src.preprocessing import clean_text

def train_models(data_path='data/WELFake_Dataset.csv'):
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    df['text'] = df['text'].astype(str).apply(clean_text)
    
    X = df['text']
    y = df['label']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Vectorizing text...")
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,3))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    print("Training models...")
    log_reg = LogisticRegression(max_iter=1000, C=3)
    rf = RandomForestClassifier(n_estimators=300, max_depth=50)
    xgb = XGBClassifier(eval_metric='logloss', n_estimators=300, max_depth=7, learning_rate=0.1)

    
    log_reg.fit(X_train_vec, y_train)
    rf.fit(X_train_vec, y_train)
    xgb.fit(X_train_vec, y_train)
    
    # Evaluate
    for name, model in zip(["Logistic Regression","Random Forest","XGBoost"], [log_reg, rf, xgb]):
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, preds))
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(tfidf, "models/tfidf.pkl")
    joblib.dump((log_reg, rf, xgb), "models/base_models.pkl")
    print("Models and vectorizer saved in /models/")

if __name__ == "__main__":
    train_models()

