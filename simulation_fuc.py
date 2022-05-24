import io
import json                                     
import logging   
import time          
from flask import Flask, request, jsonify, abort, make_response
app = Flask(__name__)          
app.logger.setLevel(logging.DEBUG)

AngleHistory = []
def simulateSensorData(boom_angle):

  global result_angle
  current_time = time.time() * 1000
  prev_AngleTime = {"boom_angle": f"{boom_angle}", "ts": f"{current_time}"}
  AngleHistory.insert(0, prev_AngleTime)
  prev_Angle = []

  ### discards all elements after 30th element ###
  AngleHistory[30:] = []

  doState = int(boom_angle) - int(AngleHistory[-1]["boom_angle"])
  #print(abs(doState))
  if abs(doState) <= 1:                                                                    # stopped
      print("stopped")
      for prev_boomAngle in AngleHistory[-3:]:
          angleDifference = int(boom_angle) - int(prev_boomAngle["boom_angle"])
          prev_Angle.append(angleDifference)
  else:                                                                                    # moved
      print("moved")
      for prev_boomAngle in AngleHistory[-5:]:
          angleDiff = int(boom_angle) - int(prev_boomAngle["boom_angle"])
          prev_Angle.append(angleDiff)

  total_sum = sum(prev_Angle)
  avgAngleDiff =  total_sum/len(prev_Angle)
  #print(avgAngleDifference)
  #print(avgAngleDifference)
  ### return value on certain condition ###
  if avgAngleDiff == 0:            # boom is still
    result_angle = boom_angle
  elif avgAngleDiff < -2:          # boom is lowering
    result_angle = boom_angle - 10
  elif avgAngleDiff > 2:           # boom is lifting
    result_angle = boom_angle + 10

  return result_angle

def publishSimulationResult(result_angle):
  current_time = time.time() * 1000
  payload = "{\r\n  \"result\" : %s,\r\n  \"ts\": %04d\r\n}\r\n" % (result_angle, current_time)
  return payload

@app.route("/sim", methods=['POST'])
def test_method():   
   print("request.data: ", request.data)      
   print("request.json: ", request.json)      
   # if not request.json or 'con' not in request.json: 
   #    abort(400)
            
   # get the base64 encoded string
   key1 = request.json['sensor1_value']
   key2 = request.json['sensor2_value']
   sensor1_rowtime = request.json['sensor1_rowtime'] 
   print("key1: ", key1)
   print("key2: ", key2)

   sim_result1 = simulateSensorData(int(key1))
   sim_result2 = simulateSensorData(int(key2))
   
   # {time:sensor1_time, input:{1_id, 1_val, 2_id, 2_val}, result:{1_res, 2_res}}
   #result = json.dumps(sim_result, ensure_ascii=False)
   #res = make_response(result)
   print("sensor1 res: " , sim_result1, type(sim_result1))
   print("sensor2 res: " , sim_result2)
   #print("time: ", sensor1_rowtime, "input: [", request.json['sensor1_id'],  request.json['sensor1_value'], request.json['sensor2_id'],  request.json['sensor2_value'],"], result: [", sim_result1 , sim_result2, "]")
   print(json.dumps({"time": sensor1_rowtime, "input": {request.json['sensor1_id']: sim_result1, request.json['sensor2_id']: sim_result2}, "result": [sim_result1 , sim_result2]}))
   print(json.dumps({request.json['sensor1_id']: sim_result1, request.json['sensor2_id']: sim_result2}))
   resultJson = json.dumps({"time": sensor1_rowtime, "input": {request.json['sensor1_id']: sim_result1, request.json['sensor2_id']: sim_result2}, "result": [sim_result1 , sim_result2]})
  
   return resultJson

  
def run_server_api():
   app.run(host='0.0.0.0', port=7972)
  
  
if __name__ == "__main__":     
   run_server_api()