import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename

# Mengaktifkan Flask
app = Flask(__name__)

# Konfigurasi folder untuk menyimpan gambar upload dan dataset
UPLOAD_FOLDER = 'uploads'
DATASET_FOLDER = 'dataset'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Membuat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Memuat model VGG16 untuk ekstraksi fitur
base_model = VGG16(weights='imagenet', include_top=False)

# Memeriksa apakah file memiliki ekstensi yang diperbolehkan.
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi untuk ekstraksi fitur
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.vgg16.preprocess_input(expanded_img_array)
    features = base_model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

# Fungsi untuk menentukan gambar serupa
def build_feature_database():
    feature_database = {}
    for img_name in os.listdir(DATASET_FOLDER):
        if allowed_file(img_name):
            img_path = os.path.join(DATASET_FOLDER, img_name)
            feature_database[img_name] = extract_features(img_path)
    return feature_database

# menampilkan hasil gambar serupa sebanyak 5 gambar
def find_similar_images(query_features, feature_database, top_n=5):
    similarities = {}
    for img_name, features in feature_database.items():
        similarity = cosine_similarity(query_features.reshape(1, -1), features.reshape(1, -1))[0][0]
        similarities[img_name] = similarity
    
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[:top_n]

# Memuat database fitur
feature_database = build_feature_database()

# Fungsi utama
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='Tidak ada bagian file')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='Tidak ada file yang dipilih')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            query_features = extract_features(file_path)
            similar_images = find_similar_images(query_features, feature_database)
            
            return render_template('results.html', query_image=filename, similar_images=similar_images)
    return render_template('index.html')

# Fungsi untuk menampilkan gambar
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Fungsi untuk menampilkan dataset
@app.route('/dataset/<filename>')
def dataset_file(filename):
    return send_from_directory(DATASET_FOLDER, filename)

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)