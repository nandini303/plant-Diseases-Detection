from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
import numpy as np
import tensorflow as tf
import json
import uuid
import os
from PIL import Image
from functools import wraps

# ================= APP CONFIG =================
app = Flask(__name__)
app.secret_key = "phasal_secret_key_2024"

UPLOAD_FOLDER = "uploadimages"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= SIMPLE USER STORE =================
# In-memory user store (replace with a DB for production)
users = {}

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(
    "models/plant_disease_recog_model_pwp.keras"
)

# ================= LABELS (MODEL ORDER) =================
labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ================= LOAD JSON (LIST) =================
with open("plant_disease.json", "r") as f:
    plant_disease = json.load(f)

# ================= LOAD FALLBACK TREATMENTS =================
with open("fallback_treatments.json", "r") as f:
    fallback_treatments = json.load(f)

# ================= HELPER FUNCTIONS =================
def get_disease_category(disease_name):
    """Determine the category of disease for fallback treatment"""
    disease_lower = disease_name.lower()
    
    if any(keyword in disease_lower for keyword in ['scab', 'rot', 'rust', 'mildew', 'blight', 'spot', 'mold']):
        return 'fungal_diseases'
    elif any(keyword in disease_lower for keyword in ['bacterial', 'bacteria']):
        return 'bacterial_diseases'
    elif any(keyword in disease_lower for keyword in ['virus', 'viral', 'mosaic', 'curl']):
        return 'viral_diseases'
    elif any(keyword in disease_lower for keyword in ['mite', 'insect', 'pest']):
        return 'pest_infestations'
    else:
        return 'fungal_diseases'  # Default to fungal

def get_related_diseases(disease_name):
    """Get related diseases for additional treatment options"""
    related = fallback_treatments.get("related_diseases", {}).get(disease_name, [])
    related_info = []
    
    for related_disease in related[:2]:  # Limit to 2 related diseases
        for disease_data in plant_disease:
            if disease_data["name"] == related_disease:
                related_info.append({
                    "name": related_disease,
                    "cure": disease_data.get("cure", ""),
                    "prescription": disease_data.get("prescription", "")
                })
                break
    
    return related_info

def enhance_disease_info(disease_info):
    """Enhance disease info with fallback treatments and related diseases"""
    disease_name = disease_info["name"]
    
    # Add fallback treatment if cure is not available or generic
    if not disease_info.get("cure") or disease_info.get("cure") == "No specific treatment information available.":
        category = get_disease_category(disease_name)
        fallback = fallback_treatments.get(category, {})
        
        disease_info["cure"] = fallback.get("general_treatment", disease_info.get("cure", ""))
        if not disease_info.get("prescription") or disease_info.get("prescription") == "No specific prescription available.":
            disease_info["prescription"] = fallback.get("prescription", "")
        if not disease_info.get("supplements"):
            disease_info["supplements"] = fallback.get("supplements", "")
    
    # Add related diseases information
    related_diseases = get_related_diseases(disease_name)
    disease_info["related_diseases"] = related_diseases
    
    # Add general prevention tips
    category = get_disease_category(disease_name)
    disease_info["prevention"] = fallback_treatments.get(category, {}).get("prevention", "")
    
    return disease_info

# ================= AUTH ROUTES =================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('home'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Username already exists.', 'error')
        else:
            users[username] = password
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# ================= ROUTES =================
@app.route('/')
@login_required
def home():
    return render_template("home.html")

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= IMAGE PREPROCESSING =================
def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def model_predict(image_path):
    img = extract_features(image_path)
    preds = model.predict(img)

    index = int(np.argmax(preds))          # INTEGER index
    confidence = float(np.max(preds)) * 100  # Get confidence percentage
    
    disease_info = plant_disease[index].copy()    # LIST access and copy to avoid modifying original
    disease_info["confidence"] = round(confidence, 2)
    
    # If confidence is low, add more fallback options
    if confidence < 70:  # Low confidence threshold
        disease_info["low_confidence"] = True
        # Add more comprehensive fallback treatments
        category = get_disease_category(disease_info["name"])
        fallback = fallback_treatments.get(category, {})
        
        # Enhance with general treatment advice
        if disease_info.get("cure"):
            disease_info["cure"] += f" \n\nNote: Due to lower prediction confidence ({confidence:.1f}%), also consider: {fallback.get('general_treatment', '')}"
    
    # Enhance with fallback treatments and related diseases
    enhanced_info = enhance_disease_info(disease_info)
    
    return enhanced_info

# ================= UPLOAD HANDLER =================
@app.route('/upload/', methods=['POST'])
@login_required
def uploadimage():

    filename = None
    save_path = None

    # ---------- CASE 1: FILE UPLOAD ----------
    if 'img' in request.files and request.files['img'].filename != "":
        image = request.files['img']
        filename = f"{uuid.uuid4().hex}_{image.filename}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(save_path)

    else:
        return redirect('/')

    # ---------- PREDICTION ----------
    prediction = model_predict(save_path)

    return render_template(
        "home.html",
        result=True,
        imagepath=f"/uploadimages/{filename}",
        prediction=prediction
    )

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)
