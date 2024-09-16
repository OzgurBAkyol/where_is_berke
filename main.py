import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)
while True: # eğer webcam başarıyla açılırsa çalışıcak
    _ , img = webcam.read() # webcamdan yakalayıp img değişkenine atadık
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    # ne kadar yüz tanıyosa hepsini faces de depoluyo
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), # sol üst nokta
                      (x+w, y+h), # sağ alt
                      (0,255,0), 3) # yüzü kırmızı çizer, 2 de çizginin kalınlığı
    # burdaki for loop ile de tüm yüzlerin etrafında birer kare çiziyo olcaz
    cv2.imshow("Where is Berke", img)
    key = cv2.waitKey(10) # 10 milisaniye
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()



# düzenlenmesi gereken kısım
# kendi yüzümü toplama
import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)
user_id = 'berke'  # Kullanıcı ID'si
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

# LBPH eğitimi
import cv2
import os
import numpy as np

# LBPH yüz tanıma modeli
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(data_folder):
    image_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
    face_samples = []
    ids = []
    for image_path in image_paths:
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        user_id = int(image_path.split('_')[0][-1])  # ID'den alınabilir
        face_samples.append(gray_image)
        ids.append(user_id)
    return face_samples, ids

# Verileri yükle ve eğit
data_folder = 'face_data'
faces, ids = get_images_and_labels(data_folder)
recognizer.train(faces, np.array(ids))

# Modeli kaydet
recognizer.write('trainer.yml')


# yüz tanıma
import cv2

# Yüklenmiş model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)
while True:
    _, img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))
        user_id, confidence = recognizer.predict(face)

        if confidence < 50:  # Güven aralığı, model ne kadar emin?
            cv2.putText(img, "Berke", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Bilinmiyor", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(10) == 27:  # ESC tuşu
        break

webcam.release()
cv2.destroyAllWindows()
