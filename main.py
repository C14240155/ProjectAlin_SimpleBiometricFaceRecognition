import os
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(image_path):
    """Mendeteksi wajah dalam gambar dan memotongnya (Crop)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print(f"   [⚠️ Skip] Tidak mendeteksi wajah pada: {os.path.basename(image_path)}")
        return None

    (x, y, w, h) = faces[0]
    cropped_face = img[y:y+h, x:x+w]
    return cropped_face


def load_custom_faces(folder_path, height=32, width=32):
    """Membaca foto asli dari folder dan mengubahnya menjadi vektor matematika."""
    images = []
    valid_filenames = [] 

    if not os.path.exists(folder_path):
        return None, None

    print("\n[INFO] Memproses gambar di dataset dengan Haar Cascade...")
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            cropped_face = detect_and_crop_face(img_path)

            if cropped_face is not None:
                # ---------------------------------------------------------
                # [KONSEP: DIMENSI (RUANG ASLI)]
                # Menetapkan bahwa setiap ruang vektor akan memiliki jumlah dimensi tetap.
                # Jika H=64 dan W=64, maka dimensinya adalah 64 x 64 = 4096.
                # ---------------------------------------------------------
                img_resized = cv2.resize(cropped_face, (width, height))

                # ---------------------------------------------------------
                # [KONSEP: EUCLIDEAN VECTOR SPACE]
                # .flatten() mengubah matriks 2D menjadi vektor 1D.
                # Titik inilah yang secara resmi memindahkan gambar dari dunia visual 
                # menjadi sebuah "titik koordinat" di dalam Ruang Vektor Euclidean dimensi tinggi.
                # ---------------------------------------------------------
                img_flattened = img_resized.flatten()

                images.append(img_flattened)
                valid_filenames.append(filename)
                print(f"   [✅ Sukses] Wajah diekstrak dari: {filename}")

    # ---------------------------------------------------------
    # [KONSEP: ROW SPACE]
    # np.array() menggabungkan semua vektor menjadi Matriks X (M x N).
    # Setiap BARIS (Row) di dalam matriks ini adalah 1 gambar wajah utuh.
    # ---------------------------------------------------------
    return np.array(images, dtype=np.float64), valid_filenames


def train_eigenfaces(X, num_components=5):
    """Proses pelatihan Eigenface menggunakan manipulasi Ruang Matriks dan Basis."""
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face

    # ---------------------------------------------------------
    # [KONSEP: HUBUNGAN ROW SPACE & COLUMN SPACE]
    # Kita tidak mencari kovarian dari N x N (piksel x piksel) karena ukurannya raksasa.
    # Kita gunakan trik (X * X^T) untuk mencari matriks berukuran M x M (jumlah gambar).
    # Ini memanfaatkan teorema bahwa eigen-value dari Row Space dan Column Space itu saling terhubung.
    # ---------------------------------------------------------
    L = np.dot(X_centered, X_centered.T)

    eigenvalues, eigenvectors_L = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors_L = eigenvectors_L[:, idx]

    # ---------------------------------------------------------
    # [KONSEP: COLUMN SPACE]
    # Mengembalikan vektor eigen (dari ruang M x M tadi) KEMBALI ke Column Space ruang aslinya (N dimensi).
    # Hasil perkalian ini menciptakan wajah hantu / Eigenfaces.
    # ---------------------------------------------------------
    eigenfaces = np.dot(X_centered.T, eigenvectors_L).T

    # ---------------------------------------------------------
    # [KONSEP: BASIS (ORTHONORMAL BASIS)]
    # Loop ini membagi vektor dengan panjang aslinya (norm) agar panjangnya menjadi tepat 1.
    # Tujuannya menciptakan Orthonormal Basis (vektor tegak lurus penyusun ruang baru bernama Face Space).
    # ---------------------------------------------------------
    for i in range(eigenfaces.shape[0]):
        norm = np.linalg.norm(eigenfaces[i])
        if norm > 0:
            eigenfaces[i] /= norm

    # ---------------------------------------------------------
    # [KONSEP: REDUKSI DIMENSI]
    # Memangkas / membuang basis yang tidak penting.
    # Kita turun dari dimensi R^4096 (jumlah piksel asli) menjadi sub-ruang R^10 saja.
    # ---------------------------------------------------------
    eigenfaces = eigenfaces[:num_components]

    # Mengalikan wajah asli dengan Basis untuk mencari posisinya (koordinat bobot) di ruang yang baru.
    weights = np.dot(X_centered, eigenfaces.T)

    return mean_face, eigenfaces, weights


def recognize_face(test_face, mean_face, eigenfaces, weights):
    """Mencocokkan wajah baru berdasarkan Jarak Euclidean di sub-ruang dimensi rendah."""
    test_centered = test_face - mean_face
    
    # Memproyeksikan wajah uji ke Basis yang baru saja dibuat
    test_weight = np.dot(test_centered, eigenfaces.T)

    # ---------------------------------------------------------
    # [KONSEP: EUCLIDEAN METRIC / DISTANCE]
    # np.linalg.norm mengukur 'jarak lurus' antara titik koordinat wajah database 
    # dan titik koordinat wajah uji di dalam sub-ruang Euclidean.
    # ---------------------------------------------------------
    distances = np.linalg.norm(weights - test_weight, axis=1)

    best_match_idx = np.argmin(distances)
    min_distance = distances[best_match_idx]

    return best_match_idx, min_distance


# --- EKSEKUSI FRAMEWORK ---
if __name__ == "__main__":
    print("=== Simple Biometric Face Recognition (Auto-Crop Haar Cascade) ===")

    H, W = 64, 64
    folder_dataset = "dataset_wajah"

    X_train, valid_filenames = load_custom_faces(folder_dataset, height=H, width=W)

    if X_train is not None and len(X_train) > 0:
        print(f"\n[INFO] Dataset berhasil dimuat!")
        print(f"Jumlah foto valid (M): {X_train.shape[0]} | Dimensi Vektor Piksel (N): {X_train.shape[1]}")

        k_dimensions = 10
        if k_dimensions > X_train.shape[0]:
            k_dimensions = X_train.shape[0]

        mean_face, eigenfaces, train_weights = train_eigenfaces(
            X_train, num_components=k_dimensions
        )
        print(f"[INFO] Training Selesai. Ruang Vektor dipangkas menjadi {eigenfaces.shape[0]} Basis Utama.")

        foto_uji = "sampleRonai.jpeg"  # Sesuaikan dengan nama file Anda

        if os.path.exists(foto_uji):
            print(f"\n[INFO] Memproses foto uji '{foto_uji}'...")
            
            cropped_test = detect_and_crop_face(foto_uji)

            if cropped_test is not None:
                img_test_resized = cv2.resize(cropped_test, (W, H))
                test_face_vector = img_test_resized.flatten().astype(np.float64)

                matched_idx, distance = recognize_face(
                    test_face_vector, mean_face, eigenfaces, train_weights
                )

                nama_file_tercocok = valid_filenames[matched_idx]

                print("\n--- Hasil Analisis Aljabar Linier ---")
                print(f"Foto '{foto_uji}' paling dekat dengan foto '{nama_file_tercocok}' di dalam database.")
                print(f"Jarak Euclidean terdekat: {distance:.2f}")
            else:
                print(f"❌ [GAGAL] Wajah tidak terdeteksi pada foto uji '{foto_uji}'. Coba foto yang lebih jelas.")
        else:
            print(f"\n[⚠️ Peringatan] File '{foto_uji}' tidak ditemukan untuk simulasi uji.")

    else:
        print("\n[⚠️ Gagal] Tidak ada wajah yang berhasil diekstrak dari folder dataset!")