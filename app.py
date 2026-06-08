import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import numpy as np
import cv2

# Import fungsi-fungsi dari skrip aslimu
from face_recognition import load_custom_faces, train_eigenfaces, detect_and_crop_face, recognize_face

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------------------------------------
# Proses Training dilakukan sekali saat server mulai berjalan
# ---------------------------------------------------------
H, W = 64, 64
folder_dataset = "dataset_wajah"
X_train, valid_filenames = load_custom_faces(folder_dataset, height=H, width=W)

if X_train is not None and len(X_train) > 0:
    mean_face, eigenfaces, train_weights = train_eigenfaces(X_train, variance_target=0.95)
    print("✅ Model siap! Server berjalan...")
else:
    print("❌ Gagal memuat dataset.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Proses gambar menggunakan algoritma PCA kamu
        cropped_test = detect_and_crop_face(filepath)
        
        if cropped_test is not None and cropped_test.size > 0:
            img_test_resized = cv2.resize(cropped_test, (W, H))
            test_face_vector = img_test_resized.flatten().astype(np.float64) / 255.0

            matched_idx, distance = recognize_face(
                test_face_vector, mean_face, eigenfaces, train_weights, threshold=15.0
            )

            # Hapus file sementara setelah diproses
            os.remove(filepath)

            if matched_idx != -1:
                nama_file_tercocok = valid_filenames[matched_idx]
                return jsonify({
                    'status': 'success',
                    'message': 'Wajah Dikenali',
                    'match': nama_file_tercocok,
                    'distance': round(distance, 2)
                })
            else:
                return jsonify({
                    'status': 'unknown',
                    'message': 'Wajah Tidak Dikenali (Unknown)',
                    'distance': round(distance, 2)
                })
        else:
            os.remove(filepath)
            return jsonify({'error': 'Wajah tidak terdeteksi pada gambar.'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)