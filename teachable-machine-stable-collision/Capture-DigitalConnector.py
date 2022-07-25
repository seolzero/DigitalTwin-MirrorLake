import json
import requests
#from PIL import Image
import numpy as np
import cv2
import base64
from cv2 import VideoCapture
from cv2 import waitKey 

#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    #h, w, _ = img.shape
    #cx = h / 2
    img = img[:, 200:200+img.shape[0]]
    img = cv2.flip(img, 1)

    ret, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    ####print(jpg_as_text[:80])
    payload = json.dumps({"data": jpg_as_text})
    #print(payload)
    cv2.imshow('img_roi', img)
    key = cv2.waitKey(1000)


    url = "http://localhost:1220/DigitalConnector/SensorGroup/CD"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response) 

    if key == 27: #esc
        break

cap.release()
cv2.destroyAllWindows()