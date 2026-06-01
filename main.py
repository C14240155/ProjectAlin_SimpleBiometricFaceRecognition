# Simple Biometric Face Recognition berbasis PCA dengan pendekatan Aljabar Linier khususnya materi Euclidean Vector Space, Row Space, Column Space, Eigenvalue, Eigenvector.
import os
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(image_path):
    #Mendeteksi wajah dalam gambar dan memotongnya (Crop).
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print(f"Tidak mendeteksi wajah pada: {os.path.basename(image_path)}")
        return None

    (x, y, w, h) = faces[0]
    cropped_face = img[y:y+h, x:x+w]
    return cropped_face


def load_custom_faces(folder_path, height=32, width=32):
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

            if cropped_face is not None:

                img_resized = cv2.resize(cropped_face, (width, height))

                img_flattened = img_resized.flatten()

                images.append(img_flattened)
                valid_filenames.append(filename)
                print(f"   [✅ Sukses] Wajah diekstrak dari: {filename}")

    return np.array(images, dtype=np.float64), valid_filenames


def train_eigenfaces(X, num_components=5):
    # Proses pelatihan Eigenface menggunakan manipulasi Ruang Matriks dan Basis.
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face

    L = np.dot(X_centered, X_centered.T)

    eigenvalues, eigenvectors_L = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors_L = eigenvectors_L[:, idx]

    eigenfaces = np.dot(X_centered.T, eigenvectors_L).T

    for i in range(eigenfaces.shape[0]):
        norm = np.linalg.norm(eigenfaces[i])
        if norm > 0:
            eigenfaces[i] /= norm

    eigenfaces = eigenfaces[:num_components]

    weights = np.dot(X_centered, eigenfaces.T)

    return mean_face, eigenfaces, weights


def recognize_face(test_face, mean_face, eigenfaces, weights):
    # Mencocokkan wajah baru berdasarkan Jarak Euclidean di sub-ruang dimensi rendah.
    test_centered = test_face - mean_face
    
    test_weight = np.dot(test_centered, eigenfaces.T)

    distances = np.linalg.norm(weights - test_weight, axis=1)

    best_match_idx = np.argmin(distances)
    min_distance = distances[best_match_idx]

    return best_match_idx, min_distance

if __name__ == "__main__":
    print("=== Simple Biometric Face Recognition berbasis PCA dengan pendekatan Aljabar Linier ===")

    H, W = 64, 64
    folder_dataset = "dataset_wajah"

    X_train, valid_filenames = load_custom_faces(folder_dataset, height=H, width=W)

    if X_train is not None and len(X_train) > 0:
        print(f"\nDataset berhasil dimuat!")
        print(f"Jumlah foto valid (M): {X_train.shape[0]} | Dimensi Vektor Piksel (N): {X_train.shape[1]}")

        k_dimensions = 10
        if k_dimensions > X_train.shape[0]:
            k_dimensions = X_train.shape[0]

        mean_face, eigenfaces, train_weights = train_eigenfaces(
            X_train, num_components=k_dimensions
        )
        print(f"Training Selesai. Ruang Vektor dipangkas menjadi {eigenfaces.shape[0]} Basis Utama.")

        foto_uji = "sampleRonai.jpeg"

        if os.path.exists(foto_uji):
            print(f"\nMemproses foto uji '{foto_uji}'...")
            
            cropped_test = detect_and_crop_face(foto_uji)

            if cropped_test is not None:
                img_test_resized = cv2.resize(cropped_test, (W, H))
                test_face_vector = img_test_resized.flatten().astype(np.float64)

                matched_idx, distance = recognize_face(
                    test_face_vector, mean_face, eigenfaces, train_weights
                )

                nama_file_tercocok = valid_filenames[matched_idx]

                print("\n--- Hasil Analisis Foto ---")
                print(f"Foto '{foto_uji}' paling dekat dengan foto '{nama_file_tercocok}' di dalam database.")
                print(f"Jarak Euclidean terdekat: {distance:.2f}")
            else:
                print(f"Wajah tidak terdeteksi pada foto uji '{foto_uji}'. Coba foto yang lebih jelas.")
        else:
            print(f"\nFile gambar uji '{foto_uji}' tidak ditemukan untuk simulasi uji.")

    else:
        print("\nTidak ada wajah yang berhasil diekstrak dari folder dataset!")