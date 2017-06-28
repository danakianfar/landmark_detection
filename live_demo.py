import numpy as np
from demo_utils import *
import matplotlib.pyplot as plt
import cv2

# Load CNN
custom_objects={'p_norm_loss': p_norm_loss, 'landmark_accuracy' : landmark_accuracy, 'landmark_loss': landmark_loss}
model_path = 'models/HeadFalse-spatial-2D-mean_squared_error-2.841274347020494-0.92591288972434949.h5'
#model_path = 'models/AugmentedHeadFalse-double_tower-2D-mean_squared_error-5.1912007235517406-0.81428417525482011.h5'
model = load_model(model_path, custom_objects)

# Load OpenCV classifiers
path = '/home/jgalle29/anaconda3/share/OpenCV/haarcascades/'
face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Or use specific image
    # frame = cv2.imread('Screen-shot-2010-04-22-at-10.07.43-AM_6.png')

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        # Plot a box around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)

        # Get the gray and color regions of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 6)

        for (ex,ey,ew,eh) in eyes:

            # If a detected eye is more than 10% of the face area, ignore it
            if ew * eh > 0.1 * w *h:
                continue

            # Resize the eye patch to feed the network
            eye_img = cv2.resize(roi_gray[ey:ey+eh, ex:ex+ew], (120, 80))

            #Is this a left eye? Then flip it to be a right eye for the CNN
            if (ex+x) < x + w/2:
                eye_img = cv2.flip(eye_img,1)

            # Put channels first and normalize by 255
            eye_img = np.reshape(eye_img, (1, 80, 120)) / 255.0

            # Forward pass
            ldmarks = model_predict(model, eye_img[np.newaxis, :, :, :], 1, 2)[0,:,:]

            #Is this a left eye? Then flip the predicted landmakrs
            if (ex+x) < x + w/2:
                ldmarks[:,0] = 120 - ldmarks[:,0]

            # Scale the landmarks according to the resize factor and displace
            # them to the eye patch origin
            ldmarks = (ldmarks * np.array([eh/120., ew/80.])) + np.array([ex, ey])

            # Draw a box around the eye
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)

            # Plot each pf the landmarks
            for ix in range(28):
                location = tuple(ldmarks[ix,:].astype(int).tolist())
                cv2.circle(roi_color,location, 1, (0,0,255))

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
