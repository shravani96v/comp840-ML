from flask import Flask, render_template, request
import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

mnist = fetch_openml('mnist_784', version=1)
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=['POST'])
def upload():
    # read file and convert string data to numpy array
    file = request.files.getlist("file")[0].read()
    img = np.fromstring(file, np.uint8)
    # convert numpy array to image
    # Image decoding
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

    ann = MLPClassifier(hidden_layer_sizes=(10, 100), max_iter=30)
    X, y = mnist['data'], mnist['target']
    y = y.astype(np.uint8)

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    ann.fit(X_train, y_train)

    # image pre processing
    img = img.ravel()
    img = img.reshape(1, 784)
    digit_prediction = ann.predict(img)

    score = ann.score(X_test, y_test) * 100

    digit_prediction = str(digit_prediction[0])

    return render_template("complete.html", prediction=digit_prediction, accuracy=score)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
