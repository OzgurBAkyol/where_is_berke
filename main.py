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

