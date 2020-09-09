import cv2
import sys
import numpy as np
import face_recognition

rectangle = False
use_face_recognition = True

cap = cv2.VideoCapture(0)
if(not use_face_recognition):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
    ret, frame = cap.read()

    if(use_face_recognition):
        faces = face_recognition.face_locations(frame);
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
    
    print("Found {0} faces!".format(len(faces)))

    if(not rectangle):
        tempImg = frame.copy()
        maskShape = (frame.shape[0], frame.shape[1], 1)
        mask = np.full(maskShape, 0, dtype=np.uint8)

    if(use_face_recognition):
        for face in faces:

            top, right, bottom, left = face

            if(rectangle):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                face = frame[top:bottom, left:right]

                (h, w) = face.shape[:2]

                kW = int(w/1.0)
                kH = int(h/1.0)

                if kW % 2 == 0:
                    kW -= 1
                if kH % 2 == 0:
                    kH -= 1

                face = cv2.GaussianBlur(face, (kW, kH) , 0)

                frame[top:bottom, left:right] = face;
            else:
                tempImg [top:bottom, left:right] = cv2.blur(tempImg [top:bottom, left:right] ,(40,40))
                cv2.ellipse(tempImg , ( ( int((left + right )/2), int((top + bottom)/2 )),(right - left,bottom - top), 0), (0, 255, 0), 5)
                cv2.ellipse(mask , ( ( int((left + right )/2), int((top + bottom)/2 )),(right - left,bottom - top), 0), 255, -1)
    else:
        for (x, y, w, h) in faces:
            if(rectangle):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face = frame[y:(y+w), x:(x+h)]

                (h, w) = face.shape[:2]

                kW = int(w/1.0)
                kH = int(h/1.0)
        
                if kW % 2 == 0:
                    kW -= 1
                if kH % 2 == 0:
                    kH -= 1

                face = cv2.GaussianBlur(face, (kW, kH) , 0)

                frame[y:(y+w), x:(x+h)] = face;
            else:
                tempImg [y:y+h, x:x+w] = cv2.blur(tempImg [y:y+h, x:x+w] ,(40,40))
                cv2.ellipse(tempImg , ( ( int((x + x + w )/2), int((y + y + h)/2 )),(w,h), 0), (0, 255, 0), 5)
                cv2.ellipse(mask , ( ( int((x + x + w )/2), int((y + y + h)/2 )),(w,h), 0), 255, -1)

    if(not rectangle):
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(frame,frame,mask = mask_inv)
        img2_fg = cv2.bitwise_and(tempImg,tempImg,mask = mask)
        frame = cv2.add(img1_bg,img2_fg)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
