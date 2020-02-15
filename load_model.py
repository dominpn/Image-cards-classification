import numpy as np
import cv2
from keras.models import load_model
from keras import Model

cards = ["ACE", "KING", "JACK", "NINE", "QUEEN", "TEN"]

def list_cards_to_str(list):
    result = ""
    for i in range(len(cards)):
        result += cards[i] + ": " + format(float(list[0][i]), ".2f") + ", "
    return result

def get_max_index(list):
    max_index = 0
    max_value = 0.0
    for i in range(len(cards)):
        if float(list[0][i]) > max_value:
            max_index = i
            max_value = float(list[0][i])
    return max_index, max_value

def get_result_text(list):
    index, value = get_max_index(list)
    text = cards[index] + " (" + str(value * 100) + "%)"
    return text


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
    image = np.expand_dims(resized, axis=0)
    result = model.predict(image)
    #print(list_cards_to_str(result))
    text = get_result_text(result)
    cv2.putText(frame, text,
                (10,10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()