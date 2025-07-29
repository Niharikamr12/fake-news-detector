Fake News Detector
A simple web-based Fake News Detection system using Machine Learning (Logistic Regression, Random Forest, XGBoost) and TF-IDF vectorization.

Introduction & Purpose
This project helps verify the authenticity of news articles using an ensemble of ML models trained on the WELFake dataset. It also provides confidence scores and an explanation of predictions using LIME.

Installation
Clone the repository
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector

Install dependencies
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
pip install -r requirements.txt

Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

Train models (if not pre-trained)
python src/model_train.py

Run the app
python app.py
Open: http://127.0.0.1:5000/

Project Structure
app.py                 # Flask application
templates/             # HTML templates
static/                # CSS & static files
src/                   # ML scripts (preprocessing, ensemble, explainability)
models/                # Saved models & vectorizer
data/                  # Dataset

Contributing
Fork the repo & create a branch.

Make changes and commit.

Open a Pull Request.

License
MIT License.
Dataset: WELFake Dataset.

