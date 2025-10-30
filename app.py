import cv2 #opencv library
from collections import deque #double-ended queue for storing points

#Emotion happy, emotion sad

#making a function to change the text that is being displayed
def put_text(img, text, org = (30,40), scale =1, color = (0,255,0), thickness = 2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def run_face():
    face_xml =cv2.data.haarcascades + "haarcascades_frontalface_default.xml"
    smile_xml = cv2.data.haarcasades + "haarcascades_smile.xml"

    face_cascade = cv2.CascadeClassifier(face_xml)
    smile_cascade = cv2.CascadeClassifier(smile_xml)

    cap = cv2.VideoCapture(0) #short for capture. Using the camera

    #error handling
    if not cap.isOpened():
        print("Could not Open Camera")
        return
    
    happy_votes = deque(maxlen=15)

    while(True):
        ok, frame = cap.read()
        if not ok:
            break
        
        frame = cv2.flip(frame, 1) #flips the image so it looks like a mirror
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=5, minSize=[120,120])

        label = "no face" # happy, sad, no face
        color = (0,0,255)
        
        for(x,y,w,h) in faces[:1]:
            cv2.rectangle(frame, [x,y], [x+w, y+h], (100,200,255), 1) # passes in x,y that comes from faces

            roi_gray =  gray[y:y+h, x:x+w] #region of interest
            roi_color = frame[y:y+h, x:x+w]

            mouth = roi_gray[h//2:h, 0, w] #bottom half of rectangle

            smiles = smile_cascade.detectMultiScale(
                mouth,
                scaleFactor=1.1,
                minNeighbors= 18, #reduces the amount of fall negatives in our model with a threshold
                minSize= [30,30] 
            ) 
        is_happy = len(smiles) > 0
        happy_votes.append(1 if is_happy else 0)

        if score >= 0.5:
            label = "happy" 
        else:
            label = "sad"

        if label  == "happy":
            color = (0,255,0)
        else:
            color = (0,255,255)

        break
    put_text(frame, f"Expression: {label}", (30,40), 1.0, color, 2)

    put_text(frame, "press q to quit", (30, frame.shape[0]-20), 0.6, (180,180,180), 1)

    cv2.imshow("Face: Happy vs Sad", frame)

