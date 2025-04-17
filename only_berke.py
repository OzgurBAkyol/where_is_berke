import cv2
import os
import numpy as np

# --- YÜZ TANIMA MODELİ (LBPH) ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# --- BERKE'Yİ ÖĞRENME (face_data klasöründen) ---
def get_images_and_labels(data_folder, user_id_prefix):
    face_samples = []
    ids = []

    for filename in os.listdir(data_folder):
        if filename.startswith(user_id_prefix) and filename.endswith('.jpg'):
            path = os.path.join(data_folder, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                face_samples.append(image)
                ids.append(1)  # Berke için ID = 1
    return face_samples, ids

# --- KLASÖR VE ID TANIMI ---
data_folder = 'face_data'
user_id = 'berke'  # Tüm berke_*.jpg dosyaları kullanılacak

# --- VERİLERİ AL VE EĞİT ---
faces, ids = get_images_and_labels(data_folder, user_id)
if len(faces) == 0:
    print("❌ Hata: 'face_data' klasöründe 'berke' ile başlayan .jpg dosyası bulunamadı.")
    exit()

recognizer.train(faces, np.array(ids))
recognizer.save('trainer.yml')  # Model kaydı

# --- MODELİ YÜKLE VE KAMERADAN TANIMA BAŞLAT ---
recognizer.read('trainer.yml')
webcam = cv2.VideoCapture(0)

print("Eğitim tamamlandı, şimdi Berkeyi tanımaya çalışıyorum... (çıkmak için ESC)")

while True:
    ret, img = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))
        predicted_id, confidence = recognizer.predict(face)

        if confidence < 50:
            label = "Berke"
            color = (0, 255, 0)
        else:
            label = "Berke degil"
            color = (0, 0, 255)

        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Berke'yi Tanıma", img)
    if cv2.waitKey(10) == 27:  # ESC tuşu ile çık
        break

webcam.release()
cv2.destroyAllWindows()
