import cv2
import numpy as np

# Load gambar
image = cv2.imread(r"C:\Users\eka pandu\Documents\.SEMESTER 6\Praktikum Machine Vision\FOTO UJI COBA\20130108_Pekerja_Proyek_Properti_3864.jpg")
if image is None:
    print("Gambar tidak ditemukan.")
    exit()

# Resize gambar agar lebih kecil jika diperlukan
image = cv2.resize(image, (640, 480))

# Ubah ke HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Range warna kuning (dapat disesuaikan)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

# Mask untuk warna kuning
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Operasi morfologi untuk menghilangkan noise
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

# Temukan kontur
contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Hitung jumlah objek
helmet_count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:  # filter objek kecil (noise)
        helmet_count += 1
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Tampilkan jumlah helm
cv2.putText(image, f"Jumlah helm kuning: {helmet_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Tampilkan hasil
cv2.imshow("Deteksi Helm Kuning", image)
cv2.imshow("Mask", mask_cleaned)
cv2.waitKey(0)
cv2.destroyAllWindows()
