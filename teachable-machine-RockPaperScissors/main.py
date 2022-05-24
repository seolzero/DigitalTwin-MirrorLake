import tensorflow.keras
import numpy as np
import cv2
import requests
import json

url = "http://localhost:1005/test"
headers = {
    'Content-Type': 'application/json'
}

model = tensorflow.keras.models.load_model('keras_model.h5')

cap = cv2.VideoCapture(0)

size = (224, 224)

classes = ['rock', 'scissors', 'paper']
rock = 'rock' 
scissors = 'scissors' 
paper = 'paper'
ret = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w, _ = img.shape
    cx = h / 2
    img = img[:, 200:200+img.shape[0]]
    img = cv2.flip(img, 1)

    img_input = cv2.resize(img, size)
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input = (img_input.astype(np.float32) / 127.0) - 1
    img_input = np.expand_dims(img_input, axis=0)

    prediction = model.predict(img_input)
    idx = np.argmax(prediction)

    cv2.putText(img, text=classes[idx], org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)

    if classes[idx] is rock:
        ret = paper 
    elif classes[idx] is scissors: 
        ret = rock 
    else: ret = scissors 
    print("Default: " +classes[idx]+ "\nwin res: "+ ret)


    payload = json.dumps({
        "name": "RockScissorsPaper",
        "res": classes[idx],
        "win res": ret
    })
    requests.request("POST", url, headers=headers, data=payload)

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break