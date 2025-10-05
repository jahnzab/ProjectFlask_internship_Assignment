"""
Multimodal Psychodermatological Disorder Detection
Flask Web Application
Author: Internship Project
Date: October 2025
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from werkzeug.utils import secure_filename
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
fusion_model = None
image_config = None
text_config = None
tokenizer_config = None
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_models():
    """Load all trained models and configurations"""
    global fusion_model, image_config, text_config, tokenizer_config
    
    try:
        # Load fusion model
        fusion_model = keras.models.load_model('models/fusion_model.h5')
        print("✓ Fusion model loaded")
        
        # Load image config
        with open('models/image_model_config.json', 'r') as f:
            image_config = json.load(f)
        print("✓ Image config loaded")
        
        # Load text config
        with open('models/text_model_config.json', 'r') as f:
            text_config = json.load(f)
        print("✓ Text config loaded")
        
        # Load tokenizer config
        with open('models/tokenizer_config.json', 'r') as f:
            tokenizer_config = json.load(f)
        print("✓ Tokenizer config loaded")
        
        print("\nAll models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


def preprocess_image(image_path):
    """Preprocess uploaded image"""
    try:
        # Read image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img_size = tuple(image_config['img_size'])
        img = cv2.resize(img, img_size)
        
        # Apply CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def preprocess_text(text):
    """Preprocess text input"""
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and lemmatize
        words = text.split()
        words = [
            lemmatizer.lemmatize(word) 
            for word in words 
            if word not in stop_words and len(word) > 2
        ]
        
        processed_text = ' '.join(words)
        
        # Convert to sequence
        word_index = tokenizer_config['word_index']
        sequence = [word_index.get(word, 1) for word in processed_text.split()]  # 1 is OOV token
        
        # Limit vocabulary
        max_words = text_config['max_words']
        sequence = [idx if idx < max_words else 1 for idx in sequence]
        
        # Pad sequence
        max_len = text_config['max_len']
        padded = pad_sequences([sequence], maxlen=max_len, padding='post', truncating='post')
        
        return padded
    
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return None


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400
        
        # Get text input
        text_input = request.form.get('text', '')
        
        if not text_input or len(text_input.strip()) < 10:
            return jsonify({'error': 'Please provide at least 10 characters of text input'}), 400
        
        # Save uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess inputs
        processed_image = preprocess_image(filepath)
        processed_text = preprocess_text(text_input)
        
        if processed_image is None or processed_text is None:
            return jsonify({'error': 'Error processing inputs'}), 500
        
        # Make prediction
        prediction = fusion_model.predict([processed_image, processed_text], verbose=0)
        
        # Get predicted class
        predicted_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_idx])
        
        # Get all class probabilities
        class_names = image_config['class_names']
        all_predictions = {
            class_names[i]: float(prediction[0][i]) 
            for i in range(len(class_names))
        }
        
        # Sort by confidence
        sorted_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
        
        # Get disorder information
        disorder_info = get_disorder_info(class_names[predicted_idx])
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'predicted_class': class_names[predicted_idx],
            'confidence': confidence,
            'all_predictions': sorted_predictions,
            'disorder_info': disorder_info
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


def get_disorder_info(disorder_name):
    """Get information about the predicted disorder"""
    disorder_database = {
        'psoriasis': {
            'description': 'A chronic autoimmune condition causing rapid skin cell buildup, often linked to stress.',
            'symptoms': ['Red patches', 'Silvery scales', 'Dry skin', 'Itching', 'Joint pain'],
            'mental_health_link': 'Stress, anxiety, and depression can trigger or worsen psoriasis flare-ups.',
            'recommendations': [
                'Consult a dermatologist for treatment options',
                'Practice stress management techniques',
                'Consider cognitive behavioral therapy',
                'Maintain regular sleep schedule',
                'Join support groups'
            ]
        },
        'eczema': {
            'description': 'An inflammatory skin condition causing itchy, red, and dry skin patches.',
            'symptoms': ['Itchy skin', 'Red patches', 'Dry skin', 'Cracked skin', 'Swelling'],
            'mental_health_link': 'Anxiety and stress can trigger eczema flare-ups and worsen symptoms.',
            'recommendations': [
                'Use prescribed topical treatments',
                'Practice relaxation techniques',
                'Avoid known triggers',
                'Keep skin moisturized',
                'Consider therapy for anxiety management'
            ]
        },
        'acne': {
            'description': 'A skin condition involving clogged pores, often exacerbated by stress hormones.',
            'symptoms': ['Pimples', 'Blackheads', 'Whiteheads', 'Cysts', 'Scarring'],
            'mental_health_link': 'Stress increases cortisol levels, which can worsen acne. Anxiety about appearance can create a cycle.',
            'recommendations': [
                'Maintain consistent skincare routine',
                'Manage stress through exercise',
                'Consider therapy if acne affects self-esteem',
                'Consult dermatologist for treatment',
                'Practice self-compassion'
            ]
        },
        'vitiligo': {
            'description': 'A condition causing loss of skin pigmentation, associated with autoimmune factors and stress.',
            'symptoms': ['White patches on skin', 'Premature graying', 'Loss of color in mucous membranes'],
            'mental_health_link': 'Can cause significant psychological distress, anxiety, and social anxiety.',
            'recommendations': [
                'Seek dermatological treatment options',
                'Consider counseling for body image concerns',
                'Join vitiligo support communities',
                'Practice self-acceptance',
                'Use sun protection'
            ]
        },
        'stress': {
            'description': 'Chronic stress can manifest in various skin conditions and worsen existing ones.',
            'symptoms': ['Skin inflammation', 'Increased sensitivity', 'Breakouts', 'Hives', 'Hair loss'],
            'mental_health_link': 'Direct connection - stress hormones directly affect skin health.',
            'recommendations': [
                'Practice daily stress management',
                'Regular exercise and meditation',
                'Maintain healthy sleep schedule',
                'Consider therapy or counseling',
                'Build support network'
            ]
        },
        'anxiety': {
            'description': 'Anxiety disorders can trigger or worsen various dermatological conditions.',
            'symptoms': ['Skin picking', 'Excessive sweating', 'Hives', 'Flushing', 'Itching'],
            'mental_health_link': 'Bidirectional - anxiety worsens skin conditions and skin issues increase anxiety.',
            'recommendations': [
                'Seek professional mental health support',
                'Practice anxiety management techniques',
                'Consider cognitive behavioral therapy',
                'Avoid skin picking behaviors',
                'Use relaxation exercises'
            ]
        },
        'depression': {
            'description': 'Depression can affect skin health through inflammation and poor self-care.',
            'symptoms': ['Neglected skincare', 'Increased inflammation', 'Poor wound healing', 'Dull skin'],
            'mental_health_link': 'Depression affects immune function and self-care behaviors, impacting skin health.',
            'recommendations': [
                'Seek mental health professional help',
                'Establish simple self-care routines',
                'Consider antidepressant therapy if appropriate',
                'Join support groups',
                'Focus on small, achievable goals'
            ]
        },
        'normal': {
            'description': 'No significant psychodermatological disorder detected.',
            'symptoms': ['Healthy skin appearance', 'No concerning mental health indicators'],
            'mental_health_link': 'Maintaining good mental health supports overall skin health.',
            'recommendations': [
                'Continue healthy lifestyle habits',
                'Practice stress management',
                'Maintain good skincare routine',
                'Regular health checkups',
                'Stay aware of changes'
            ]
        }
    }
    
    return disorder_database.get(disorder_name.lower(), {
        'description': 'Information not available for this condition.',
        'symptoms': [],
        'mental_health_link': 'Consult healthcare professionals for accurate information.',
        'recommendations': ['Seek professional medical advice']
    })


@app.route('/metrics')
def metrics():
    """Display model metrics dashboard"""
    try:
        # Load metrics from training
        metrics_data = {
            'image_model': {},
            'text_model': {},
            'fusion_model': {}
        }
        
        # Try to load saved metrics
        if os.path.exists('models/metrics.json'):
            with open('models/metrics.json', 'r') as f:
                metrics_data = json.load(f)
        
        return render_template('metrics.html', metrics=metrics_data)
    
    except Exception as e:
        return render_template('metrics.html', metrics={}, error=str(e))


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File is too large. Maximum size is 16MB'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500


if __name__ == '__main__':
    print("="*70)
    print("MULTIMODAL PSYCHODERMATOLOGICAL DISORDER DETECTION")
    print("Flask Web Application")
    print("="*70)
    
    # Load models
    if load_models():
        print("\n✓ Application ready!")
        print("\nStarting server...")
        print("Access the application at: http://localhost:5000")
        print("\nPress Ctrl+C to stop the server")
        print("="*70)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Failed to load models. Please train the models first.")
        print("Run: python train_image_model.py")
        print("Run: python train_text_model.py")
        print("Run: python train_fusion_model.py")