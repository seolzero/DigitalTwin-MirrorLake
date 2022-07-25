import json
import requests
#from PIL import Image
import numpy as np
import cv2
import base64

print("hello") 
cap = cv2.VideoCapture(0)
print(cap.isOpened())
while cap.isOpened():
    
    print("helloooo") 
    ret, img = cap.read()
    if not ret:
        break

    key = cv2.waitKey(3000)

    if key == 27: #esc
        break

cap.release()
cv2.destroyAllWindows()

