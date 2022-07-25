import io
import json                    
import base64                  
import logging             
import numpy as np
from PIL import Image
import cv2
import requests
from flask import Flask, request, jsonify, abort
import tensorflow.keras
app = Flask(__name__)          
app.logger.setLevel(logging.DEBUG)

url = "http://localhost:1212/test"
headers = {
    'Content-Type': 'application/json'
}

model = tensorflow.keras.models.load_model('keras_model.h5')

size = (224, 224)

classes = ['rock', 'paper', 'scissors', 'what'] 
rock = 'rock' 
scissors = 'scissors' 
paper = 'paper'

@app.route("/RSP", methods=['POST'])
def test_method():         
   # print(request.json)      
   if not request.json or 'sensor1_value' not in request.json: 
      abort(400, 'json key not exist')
            
   # get the base64 encoded string
   im_b64 = request.json['sensor1_value']
   rowtime = request.json['sensor1_rowtime']
   
   str_new = im_b64 + '=' * (4 - len(im_b64) % 4)
   
   # convert it into bytes  
   img_bytes = base64.b64decode(str_new)

   # convert bytes data to PIL Image object
   try:
      img = Image.open(io.BytesIO(img_bytes))
   except:
      print("cannot identify image file")

   # PIL image object to numpy array
   img_arr = np.asarray(img)      
   #print('img shape', img_arr.shape)

   # process your img_arr here    
   img_input = cv2.resize(img_arr, size)
   img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
   img_input = (img_input.astype(np.float32) / 127.0) - 1
   img_input = np.expand_dims(img_input, axis=0)

   prediction = model.predict(img_input)
   idx = np.argmax(prediction)
   #print(classes[idx])
   #cv2.putText(img_arr, text=classes[idx], org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
   #cv2.imshow('result', img_arr)
   #cv2.waitkey(0)

   if classes[idx] is rock:
      ret = paper 
   elif classes[idx] is scissors: 
      ret = rock 
   elif classes[idx] is paper: 
      ret = scissors
   else: ret = 'what' 
   print("Default: " +classes[idx]+ "\nwin res: "+ ret)


   payload = json.dumps({
      "name": "RockScissorsPaper",
      "res": classes[idx],
      "win res": ret
   })
   #requests.request("POST", url, headers=headers, data=payload)

   result_dict = {'input': { request.json['sensor1_id']:classes[idx]}, 'result': { request.json['sensor1_id']:ret}, 'rowtime': rowtime}
   #print("result_dict: " + classes[idx])
   return result_dict
  
  
def run_server_api():
   app.run(host='0.0.0.0', port=8080)
  
  
if __name__ == "__main__":     
   run_server_api()