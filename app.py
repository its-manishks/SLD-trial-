from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load your trained model
model = load_model('./Model/keras_model.h5')  # Update the path if necessary

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['frame']
    # Decode the base64 frame
    img_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Preprocess the image
    resized_img = cv2.resize(img, (224, 224)) / 255.0  # Adjust size per your model
    resized_img = np.expand_dims(resized_img, axis=0)

    # Get predictions
    prediction = model.predict(resized_img)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
