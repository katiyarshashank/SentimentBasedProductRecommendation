# Required Package
from flask import Flask, request, render_template
import model

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['username']

    # User recommendation model call
    flag, prods = model.recommendation(user_input)

    if flag:
        # Sentiment model call
        data = model.sentiment(prods)
        return render_template('index.html', tables=[data.to_html(classes='data', index=False)])
    else:
        # User error message
        return render_template('index.html', message=prods)


if __name__ == "__main__":
    app.run()
