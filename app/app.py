from flask import Flask, render_template, request
from src.ensemble import predict_ensemble
from src.explain import explain_prediction

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        news_text = request.form['news']
        prediction, confidence = predict_ensemble([news_text])
        explanation = explain_prediction(news_text)
        return render_template('result.html',
                               prediction=prediction[0],
                               confidence=round(confidence[0]*100,2),
                               explanation=explanation)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
