import numpy as np
import cv2
from keras.models import load_model
from keras import Model

print("loading model")
model: Model = load_model('cards_model.hdf5')
print("model is loaded")
# model.predict()

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    dim = (224, 224)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    model.predict(resized)

    # Display the resulting frame
    cv2.imshow('frame',resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()