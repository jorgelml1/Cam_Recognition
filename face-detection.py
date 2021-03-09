import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")

labels = {"person_name": 1}
with open("face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}



cap = cv2.VideoCapture(0) #numero de cámara

while(True):
    #captura el frame
  ret, frame = cap.read()
  cv2.normalize(frame, frame, 0, 300, cv2.NORM_MINMAX)
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors = 4)
  for(x,y,w,h) in faces:
      print(x,y,w,h)
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = frame[y:y+h, x:x+w]

      #recognize? deep learned model predict keras tensorflow pytorch scikit learned
      id_, conf = recognizer.predict(roi_gray)
      if conf>=4 and conf <= 85:
    		#print(5: #id_)
    		#print(labels[id_])
    	       font = cv2.FONT_HERSHEY_SIMPLEX
    	       name = labels[id_]
    	       color = (255, 255, 255)
    	       stroke = 2
    	       cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


      img_item = "my-image.png"
      cv2.imwrite(img_item, roi_gray)

      color = (255,0,0) #BGR 0-255
      stroke = 2
      end_cord_x = x+w
      end_cord_y = y+h
      cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)

    #Display la imagen
  cv2.imshow('frame',frame)
  if cv2.waitKey(20) & 0xFF == ord('q'): #Terminar cunado presione q
    break

cap.release() #liberar la camara
cv2.destroyAllWindows() #Destruir la ventana
