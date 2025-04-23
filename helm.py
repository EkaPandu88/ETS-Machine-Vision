import cv2
import numpy as np
import os

# Path ke gambar
image_path = r"C:\Users\eka pandu\Documents\.SEMESTER 6\Praktikum Machine Vision\ETS Machine Vision\d.jpg"

# Load gambar
image = cv2.imread(image_path)
if image is None:
    print("Gambar tidak ditemukan.")
    exit()

# Resize gambar
image = cv2.resize(image, (640, 480))

# Ubah ke HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Range warna kuning
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

# Mask warna kuning
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Bersihkan noise
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

# Temukan kontur
contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Hitung helm
helmet_count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        helmet_count += 1
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Tambahkan teks jumlah helm
cv2.putText(image, f"Jumlah helm kuning: {helmet_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Buat folder output jika belum ada
output_dir = "hasil_deteksi"
os.makedirs(output_dir, exist_ok=True)

# Simpan hasil deteksi dan masking
cv2.imwrite(os.path.join(output_dir, "deteksi_helm.jpg"), image)
cv2.imwrite(os.path.join(output_dir, "masking_helm.jpg"), mask_cleaned)

# Tampilkan hasil
cv2.imshow("Deteksi Helm Kuning", image)
cv2.imshow("Masking", mask_cleaned)
cv2.waitKey(0)
cv2.destroyAllWindows()
