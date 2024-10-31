import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the model
model = load_model(r'C:\Users\haris\Downloads\spotgopi\test1.keras')

# Set up the class names based on your training data
sports_classes = list(os.listdir('dataset/train'))

def classify_sport_image(img):
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Rescale
    predictions = model.predict(img_array)
    return predictions

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    uploaded_file = request.files['file']
    
    if uploaded_file.filename == '':
        return "No selected file"
    
    img_path = os.path.join('static', uploaded_file.filename)
    uploaded_file.save(img_path)

    # Load and classify the image
    img = Image.open(img_path)
    preds = classify_sport_image(img)
    class_index = np.argmax(preds, axis=1)[0]
    class_name = sports_classes[class_index]
    confidence = preds[0][class_index] * 100  # Confidence percentage
    confidence_str = f"{confidence:.2f}"  # Format confidence to 2 decimal places

    return render_template('result.html', class_name=class_name, confidence=confidence_str, image_file=uploaded_file.filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
