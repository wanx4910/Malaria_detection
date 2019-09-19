import numpy as np
import pickle
from flask import Flask, render_template, request
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import imageio 

app = Flask(__name__)

loaded_model = load_model('malariaCNN.h5')
loaded_model._make_predict_function()

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

def ClassPredictor(file):
    result = loaded_model.predict(file)
    return result[0]

def process_image(img):
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    result = ClassPredictor(img)
    return result

@app.route('/result', methods = ['POST'])
def result():
    prediction=''
    if request.method == 'POST':
        file = request.files['file']
        img = np.array(image.load_img(file, target_size=(64,64)))
        result = process_image(img)
        print("result from model", result)
        if float(result)<0.5:
            prediction = 'Infected cell'
        else:
            prediction = 'Non-infected cell'
        print(prediction)
        return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run()
