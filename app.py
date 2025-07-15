from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("model/mask_detector.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        path = "static/output.jpg"
        image.save(path)
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img / 255.0, axis=0)
        prediction = model.predict(img)[0][0]
        label = "With Mask" if prediction < 0.5 else "Without Mask"
        return render_template("index.html", label=label, image=path)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
