# Simple Biometric Face Recognition berbasis PCA dengan pendekatan Aljabar Linier
import os
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    img = cv2.equalizeHist(img)

    # Kembalikan parameter ke angka yang agak aman agar tidak terlalu banyak "sampah"
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    
    if len(faces) == 0:
        return None

    # [PERBAIKAN KUNCI]: Urutkan hasil deteksi berdasarkan Area (Lebar x Tinggi) dari yang terbesar ke terkecil
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    
    # Selalu ambil wajah yang paling besar (indeks 0 setelah diurutkan)
    (x, y, w, h) = faces[0]
    
    margin = int(w * 0.1)
    y_start = max(0, y + margin)
    y_end = min(img.shape[0], y + h - margin)
    x_start = max(0, x + margin)
    x_end = min(img.shape[1], x + w - margin)
    
    cropped_face = img[y_start:y_end, x_start:x_end]
    return cropped_face

def load_custom_faces(folder_path, height=64, width=64):
    # Membaca foto asli dari folder dan mengubahnya menjadi vektor matematika.
    images = []
    valid_filenames = [] 

    if not os.path.exists(folder_path):
        return None, None

    print("\nMemproses gambar di dataset dengan Haar Cascade...")
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            cropped_face = detect_and_crop_face(img_path)

            if cropped_face is not None and cropped_face.size > 0:
                img_resized = cv2.resize(cropped_face, (width, height))
                # Standarisasi nilai vektor dari 0-255 menjadi 0-1 agar kalkulasi matriks lebih stabil
                img_flattened = img_resized.flatten() / 255.0 
                
                images.append(img_flattened)
                valid_filenames.append(filename)
                print(f"   [✅ Sukses] Wajah diekstrak dari: {filename}")
            else:
                print(f"   [❌ Gagal] Wajah tidak layak/tidak terdeteksi di: {filename}")

    return np.array(images, dtype=np.float64), valid_filenames

def train_eigenfaces(X, variance_target=0.95):
    # Proses pelatihan Eigenface menggunakan manipulasi Ruang Matriks dan Basis.
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face

    # Turk-Pentland trick: L = A * A^T (Ukuran M x M)
    L = np.dot(X_centered, X_centered.T)

    eigenvalues, eigenvectors_L = np.linalg.eigh(L)
    
    # Urutkan dari Eigenvalue terbesar (informasi fitur paling penting)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors_L = eigenvectors_L[:, idx]

    # [PERBAIKAN 4]: Menentukan jumlah K (Basis Utama) berdasarkan target Variance (misal: 95% informasi)
    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues) / total_variance
    k_components = np.argmax(cumulative_variance >= variance_target) + 1
    
    # Transformasi kembali ke ukuran N (piksel) untuk mendapatkan Eigenfaces yang sebenarnya
    eigenfaces = np.dot(X_centered.T, eigenvectors_L[:, :k_components]).T

    # Normalisasi vektor basis menjadi unit vector (panjang = 1) di Euclidean Space
    for i in range(eigenfaces.shape[0]):
        norm = np.linalg.norm(eigenfaces[i])
        if norm > 0:
            eigenfaces[i] /= norm

    # Proyeksikan data training ke Ruang Kolom Eigenfaces yang baru
    weights = np.dot(X_centered, eigenfaces.T)

    return mean_face, eigenfaces, weights

def recognize_face(test_face, mean_face, eigenfaces, weights, threshold=15.0):
    # Mencocokkan wajah baru berdasarkan Jarak Euclidean di sub-ruang dimensi rendah.
    test_centered = test_face - mean_face
    
    # Proyeksikan wajah uji ke ruang eigen
    test_weight = np.dot(test_centered, eigenfaces.T)

    # Hitung Jarak Euclidean antara wajah uji dan semua wajah di database
    distances = np.linalg.norm(weights - test_weight, axis=1)

    best_match_idx = np.argmin(distances)
    min_distance = distances[best_match_idx]

    # [PERBAIKAN 5]: Tolak kecocokan jika Jarak Euclidean lebih besar dari Threshold
    if min_distance > threshold:
        return -1, min_distance

    return best_match_idx, min_distance

if __name__ == "__main__":
    print("=== Simple Biometric Face Recognition berbasis PCA dengan pendekatan Aljabar Linier ===")

    H, W = 64, 64
    folder_dataset = "dataset_wajah"

    X_train, valid_filenames = load_custom_faces(folder_dataset, height=H, width=W)

    if X_train is not None and len(X_train) > 0:
        print(f"\nDataset berhasil dimuat!")
        print(f"Jumlah foto valid (M): {X_train.shape[0]} | Dimensi Vektor Piksel (N): {X_train.shape[1]}")

        # Training dan secara otomatis mengambil sekian basis yang mewakili 95% varian wajah
        mean_face, eigenfaces, train_weights = train_eigenfaces(X_train, variance_target=0.95)
        print(f"Training Selesai. Ruang Vektor dipangkas menjadi {eigenfaces.shape[0]} Basis Utama yang esensial.")

        foto_uji = "sampleBer.jpeg"

        if os.path.exists(foto_uji):
            print(f"\nMemproses foto uji '{foto_uji}'...")
            
            cropped_test = detect_and_crop_face(foto_uji)

            if cropped_test is not None and cropped_test.size > 0:
                img_test_resized = cv2.resize(cropped_test, (W, H))
                # Jangan lupa dibagi 255.0 sama seperti data training
                test_face_vector = img_test_resized.flatten().astype(np.float64) / 255.0

                # Parameter threshold mungkin perlu kamu sesuaikan (naikkan/turunkan) tergantung dataset
                matched_idx, distance = recognize_face(
                    test_face_vector, mean_face, eigenfaces, train_weights, threshold=15.0
                )

                print("\n--- Hasil Analisis Foto ---")
                if matched_idx != -1:
                    nama_file_tercocok = valid_filenames[matched_idx]
                    print(f"✅ Foto '{foto_uji}' DIKENALI sebagai '{nama_file_tercocok}'.")
                    print(f"Jarak Euclidean: {distance:.2f}")
                else:
                    print(f"❌ Wajah TIDAK DIKENALI (Unknown).")
                    print(f"Wajah terdekat berjarak {distance:.2f}, melebihi batas toleransi yang diizinkan.")
            else:
                print(f"Wajah tidak terdeteksi secara proporsional pada foto uji '{foto_uji}'.")
        else:
            print(f"\nFile gambar uji '{foto_uji}' tidak ditemukan untuk simulasi uji.")

    else:
        print("\nTidak ada wajah yang berhasil diekstrak dari folder dataset!")