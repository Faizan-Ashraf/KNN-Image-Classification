from flask import Flask, render_template, request
import os
from knn import KNNClassifier
from PIL import Image

app = Flask(__name__)

# Load and train the model once
knn = KNNClassifier(k=3, image_size=(32, 32))
knn.fit('./Train_Model/')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_path = os.path.join('./uploads', file.filename)
            file.save(img_path)
            prediction = knn.predict(img_path)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
