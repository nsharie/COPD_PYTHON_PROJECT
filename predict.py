# =====================================
# COPD Detection - Single Image Predict
# =====================================

import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image

# =====================================
# Load Model
# =====================================

if not os.path.exists('copd_model.h5'):
    print("ERROR: copd_model.h5 not found!")
    print("Please run train.py first.")
    exit()

print("Loading model...")
model = tf.keras.models.load_model('copd_model.h5')
print("Model loaded successfully!")

# =====================================
# Load & Preprocess Image
# =====================================

img_path = 'test_xray.jpg'   # Change this to your image path

if not os.path.exists(img_path):
    print(f"ERROR: Image not found at '{img_path}'")
    print("Please provide a valid image path.")
    exit()

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# =====================================
# Predict
# =====================================

prediction = model.predict(img_array)
confidence = float(prediction[0][0]) * 100

print("\n========== RESULT ==========")
if prediction[0][0] > 0.5:
    print(f"Diagnosis : COPD Detected")
    print(f"Confidence: {confidence:.2f}%")
else:
    print(f"Diagnosis : No COPD Detected (Normal)")
    print(f"Confidence: {100 - confidence:.2f}%")
print("============================")