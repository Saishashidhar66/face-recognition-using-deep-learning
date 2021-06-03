
import cv2
import numpy as np
from pathlib import Path

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier(r'E:\facerec-using-TLM\haarcascade_frontalface_default.xml')
print("Enter the id  of the person: ")
userId = input()
print("Enter the  name of the person: ")
userName = input()

def saveImage(img, userName, userId ):
    # Create a folder with the name as userName
    Path("dataset/{}".format(userName)).mkdir(parents=True, exist_ok=True)
    # Save the images inside the previously created folder
    cv2.imwrite("dataset/{}/{}_{}.jpg".format(userName,userName[-3:],userId), face)
    print("[INFO] Image {} has been saved in folder : {}".format(
        userId, userName))
# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]
        coords = [x, y, w, h]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    cv2.imshow('test frame',frame)
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))
        #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        saveImage(userName, userId, count)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2) #0,255,0 is green color 2 is thickness
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 1020: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")


# In[ ]:





