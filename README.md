1. Fake News Detector
A web-based Fake News Detection system using Machine Learning (Logistic Regression, Random Forest, XGBoost) with TF-IDF vectorization and LIME explainability.


2. Introduction & Purpose
Fake news spreads rapidly across digital platforms, making it difficult for users to distinguish between real and false information.This project aims to help users verify the authenticity of news articles using an ensemble of ML models trained on the WELFake dataset. It also provides explainable AI insights using LIME, showing which words influenced the prediction.

3. Key Features:

1.Ensemble of Logistic Regression, Random Forest, and XGBoost models.

2.Confidence score visualization using Chart.js.

3.Explainable AI (LIME) to show word influence.

4.Interactive web interface built with Flask.


4. Project Structure

app.py                 # Flask application
templates/             # HTML templates
static/                # CSS & static files
src/                   # ML scripts (preprocessing, ensemble, explainability)
models/                # Saved models & vectorizer
data/                  # Dataset



5. License
MIT License.
Dataset: WELFake Dataset.



