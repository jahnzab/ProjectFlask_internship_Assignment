
# import os
# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import tensorflow as tf
# import json
# import google.generativeai as genai
# import os
# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from fusion_model import (
#     LateFusionPredictor, IMAGE_CLASSES, TEXT_CLASSES,
#     TEXT_MODEL_CONFIG_PATH, TOKENIZER_CONFIG_PATH,
#     TEXT_MODEL_WEIGHTS_PATH, IMAGE_MODEL_PATH
# )

# # -------------------------
# # Flask App Setup
# # -------------------------
# app = Flask(__name__)

# # -------------------------
# # Load Configs & Tokenizer
# # -------------------------
# with open(TEXT_MODEL_CONFIG_PATH, 'r') as f:
#     text_config = json.load(f)

# with open(TOKENIZER_CONFIG_PATH, 'r') as f:
#     cfg = json.load(f)

# tokenizer = tf.keras.preprocessing.text.Tokenizer(
#     num_words=cfg.get('num_words', 10000),
#     oov_token="<OOV>"
# )
# tokenizer.word_index = cfg.get('word_index', {})

# # -------------------------
# # Late Fusion Predictor
# # -------------------------
# predictor = LateFusionPredictor(
#     image_model_path=IMAGE_MODEL_PATH,
#     text_model_path=TEXT_MODEL_WEIGHTS_PATH,
#     tokenizer=tokenizer,
#     text_config=text_config
# )

# # -------------------------
# # Gemini Setup
# # -------------------------
# GEMINI_API_KEY = "AIzaSyDsUHGkjhASXBHLmUpIUq4JlokU9K90uTs"
# genai.configure(api_key=GEMINI_API_KEY)
# gemini_model = genai.GenerativeModel('gemini-2.0-flash')
# # -------------------------
# # Flask Routes
# # -------------------------
# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     if 'image' not in request.files or 'text' not in request.form:
#         return jsonify({"success": False, "error": "Provide both image and text"}), 400

#     img_file = request.files['image']
#     text_input = request.form['text']

#     img_path = f"temp_{img_file.filename}"
#     img_file.save(img_path)

#     try:
#         # -------------------------
#         # Late Fusion Prediction
#         # -------------------------
#         mental_state_label = predictor.predict(img_path, text_input)

#         # Image model prediction
#         img_array = predictor.preprocess_image(img_path)
#         img_pred = predictor.image_model.predict(img_array, verbose=0)
#         skin_label = IMAGE_CLASSES[np.argmax(img_pred)]

#         # Optional: probabilities for frontend
#         all_preds_probs = {cls: float(np.random.rand()) for cls in TEXT_CLASSES}
#         total = sum(all_preds_probs.values())
#         all_preds_probs = {k: v/total for k,v in all_preds_probs.items()}

        
#         # -------------------------
#         # Gemini Explanation
#         # -------------------------
#         prompt = (
#         f"The person has a mental state: {mental_state_label} "
#         f"and skin condition: {skin_label}. "
#         "Give a professional, empathetic, and strong explanation with advice."
#         )


#         # Simple approach without generation config
#         gemini_response = gemini_model.generate_content(prompt)
#         explanation = gemini_response.text  
#         # -------------------------
#         # JSON Response
#         # -------------------------
#         output = {
#             "success": True,
#             "predicted_class": mental_state_label,
#             "confidence": float(np.max(list(all_preds_probs.values()))),
#             "all_predictions": all_preds_probs,
#             "disorder_info": {
#                 "description": f"The person has a mental state: {mental_state_label} and skin condition: {skin_label}.",
#                 "symptoms": ["Stress", "Anxiety", "Sleep disturbance"],  # example
#                 "mental_health_link": "This condition may relate to stress and psychological factors.",
#                 "recommendations": [explanation]
#             }
#         }

#         return jsonify(output)

#     finally:
#         if os.path.exists(img_path):
#             os.remove(img_path)

# # -------------------------
# # Run Flask
# # -------------------------
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)

import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import google.generativeai as genai

from fusion_model import (
    LateFusionPredictor, IMAGE_CLASSES, TEXT_CLASSES,
    TEXT_MODEL_CONFIG_PATH, TOKENIZER_CONFIG_PATH,
    TEXT_MODEL_WEIGHTS_PATH, IMAGE_MODEL_PATH
)

# -------------------------
# Flask App Setup
# -------------------------
app = Flask(__name__)

# -------------------------
# Load Configs & Tokenizer
# -------------------------
with open(TEXT_MODEL_CONFIG_PATH, 'r') as f:
    text_config = json.load(f)

with open(TOKENIZER_CONFIG_PATH, 'r') as f:
    cfg = json.load(f)

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=cfg.get('num_words', 10000),
    oov_token="<OOV>"
)
tokenizer.word_index = cfg.get('word_index', {})

# -------------------------
# Late Fusion Predictor
# -------------------------
predictor = LateFusionPredictor(
    image_model_path=IMAGE_MODEL_PATH,
    text_model_path=TEXT_MODEL_WEIGHTS_PATH,
    tokenizer=tokenizer,
    text_config=text_config
)

# -------------------------
# Gemini Setup
# -------------------------
GEMINI_API_KEY = "AIzaSyDsUHGkjhASXBHLmUpIUq4JlokU9K90uTs"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# -------------------------
# Flask Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files or 'text' not in request.form:
        return jsonify({"success": False, "error": "Provide both image and text"}), 400

    img_file = request.files['image']
    text_input = request.form['text']

    img_path = f"temp_{img_file.filename}"
    img_file.save(img_path)

    try:
        # -------------------------
        # Late Fusion Prediction with ACTUAL PROBABILITIES
        # -------------------------
        # Get the predicted label
        mental_state_label = predictor.predict(img_path, text_input)
        
        # Calculate ACTUAL probabilities (not random!)
        img_array = predictor.preprocess_image(img_path)
        img_pred = predictor.image_model.predict(img_array, verbose=0)
        skin_label = IMAGE_CLASSES[np.argmax(img_pred)]

        seq = predictor.tokenizer.texts_to_sequences([text_input])
        padded = pad_sequences(seq, maxlen=200, padding='post')
        text_pred = predictor.text_model.predict(padded, verbose=0)

        # Calculate fused probabilities (same logic as in predictor)
        fused = predictor.img_weight * img_pred[:, :len(TEXT_CLASSES)] + predictor.text_weight * text_pred
        
        # Apply softmax to get proper probabilities that sum to 1
        fused_probs = tf.nn.softmax(fused[0]).numpy()
        
        # Create probability dictionary with actual model predictions
        all_preds_probs = {TEXT_CLASSES[i]: float(fused_probs[i]) for i in range(len(TEXT_CLASSES))}
        
        # Get confidence for the predicted class
        confidence = all_preds_probs[mental_state_label]
        
        # -------------------------
        # Gemini Explanation
        # -------------------------
        prompt = (
            f"The person has a mental state: {mental_state_label} "
            f"and skin condition: {skin_label}. "
            "Give a professional, empathetic, and strong explanation with advice."
        )

        gemini_response = gemini_model.generate_content(prompt)
        explanation = gemini_response.text  
        
        # -------------------------
        # JSON Response
        # -------------------------
        output = {
            "success": True,
            "predicted_class": mental_state_label,
            "confidence": confidence,  # Use ACTUAL confidence
            "all_predictions": all_preds_probs,  # Use ACTUAL probabilities
            "disorder_info": {
                "description": f"The person has a mental state: {mental_state_label} and skin condition: {skin_label}.",
                "symptoms": ["Stress", "Anxiety", "Sleep disturbance"],
                "mental_health_link": "This condition may relate to stress and psychological factors.",
                "recommendations": [explanation]
            }
        }

        return jsonify(output)

    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

# -------------------------
# Run Flask
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
