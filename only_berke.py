import cv2
import os
import numpy as np

# LBPH yüz tanıma modeli
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(data_folder, user_id):
    image_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith(user_id)]
    face_samples = []
    ids = []
    for image_path in image_paths:
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        face_samples.append(gray_image)
        ids.append(1)  # ID olarak sadece "1" belirliyoruz, çünkü yalnızca sizin yüzünüz olacak.
    return face_samples, ids

# Yüz verilerini toplama kısmı
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)
user_id = 'berke'  # Kendi kullanıcı ID'niz
count = 0
data_folder = 'face_data'

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

while count < 100:  # 100 fotoğraf çek
    _, img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        file_path = os.path.join(data_folder, f'{user_id}_{count}.jpg')
        cv2.imwrite(file_path, face)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imshow("Collecting Face Data", img)
    if cv2.waitKey(10) == 27:  # ESC tuşuna basıldığında çıkış
        break
    elif count >= 100:
        break

webcam.release()
cv2.destroyAllWindows()

# Verileri yükle ve eğit
faces, ids = get_images_and_labels(data_folder, user_id)
recognizer.train(faces, np.array(ids))

# Modeli kaydet
recognizer.write('trainer.yml')

# Yüz tanıma
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer.read('trainer.yml')

webcam = cv2.VideoCapture(0)
while True:
    _, img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))
        user_id_predicted, confidence = recognizer.predict(face)

        if confidence < 50:  # Güven aralığı
            cv2.putText(img, "Berke", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Berke değil sanırım.", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(10) == 27:  # ESC tuşu
        break

webcam.release()
cv2.destroyAllWindows()