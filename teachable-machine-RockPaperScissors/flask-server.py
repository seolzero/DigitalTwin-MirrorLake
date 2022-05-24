from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
import io
from PIL import Image

app = Flask(__name__)

#get
@app.route('/echo/<param>')
def get_echo(param):
    return jsonify({"param": param})

#post
@app.route('/DigitalConnector/sensor', methods=['POST'])
def post_echo():
   data = request.get_json()
   base64_string = data['img']
   #print(imgJson)
   imgdata = base64.b64decode(base64_string)
   #dataBytesIO = io.BytesIO(imgdata)

   cv2.imshow('send_img', imgdata)

   return data

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1209)



   #cv2.imshow('send_img', jpg_original)
   #jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
   #image_buffer = cv2.imdecode(jpg_as_np, flags=1)
   #print("image_buffer: " + image_buffer)