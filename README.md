# 🏅 Sports Image Classification using Deep Learning

This project focuses on classifying sports images into six categories — **Badminton**, **Basketball**, **Cricket**, **Rugby**, **Tennis**, and **Volleyball** — using a deep learning model based on **MobileNetV3**, enhanced with a custom **Battle Royale Optimization** algorithm for hyperparameter tuning.

## 🚀 Key Features

- Lightweight and efficient MobileNetV3 architecture  
- Transfer learning with ImageNet weights  
- Custom hyperparameter tuning using Battle Royale Optimization  
- Flask-based web app for image upload and prediction  
- Achieved ~96.57% accuracy on test data  

## 📁 Dataset

- Images extracted from YouTube sports videos  
- Over 10,000 images across 6 sports  
- Preprocessed, resized to 224x224, and class-balanced  

## ⚙️ Technologies Used

- Python, TensorFlow, Keras  
- OpenCV, NumPy, Matplotlib  
- Flask (for deployment)  
- HTML, CSS (UI)  

## 💡 How to Use

1. Clone the repo  
2. (Optional) Train the model: `python train_model.py`  
3. Run the web app: `python app.py`  
4. Open browser at `http://localhost:5000` and upload a sports image  

## 📊 Results

- **Training Accuracy**: 98.03%  
- **Validation Accuracy**: 98.01%  
- **Test Accuracy**: ~96.57%

## 📌 Future Scope

- Real-time video classification  
- Cloud deployment  
- Mobile app integration  

---

**Developed by:** Gopika Raj  
_Mar Athanasius College of Engineering_  
Under the guidance of **Prof. Shinu S Kurian**
