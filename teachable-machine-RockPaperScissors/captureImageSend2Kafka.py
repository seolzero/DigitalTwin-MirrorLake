import json
import requests
#from PIL import Image
import numpy as np
import cv2
import base64
from kafka import KafkaProducer
from json import dumps 


producer = KafkaProducer(acks=0, 
compression_type='gzip', 
bootstrap_servers=['10.252.73.37:9092'],
value_serializer=lambda x: dumps(x).encode('utf-8')) 

topic = 'capimage'

cap = cv2.VideoCapture(0)
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
   payload = json.dumps({"image": jpg_as_text})
   #print(payload)
   cv2.imshow('img_roi', img)
   key = cv2.waitKey(5000)

   producer.send(topic, value=payload) 
   producer.flush() 
    

   if key == 27: #esc
      break

cap.release()
cv2.destroyAllWindows()