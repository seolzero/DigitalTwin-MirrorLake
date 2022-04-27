# import requests


import paho.mqtt.client as mqtt
import json
import time
import pandas as pd

topic = "dp_do_data"
MQTT_BROKER = "192.168.0.100"
AngleHistory = []
HeadingHistory = []
# pd.read_csv('sensor.csv')
oldAngle = ""

def ExtractSensorData(payload):
  global oldAngle

  # get "name" of DO
  DOName = payload["name"]
  # get "sensor"
  jsonSensorData = payload["sensor"]
  #print(jsonSensorData)

  ### get "angle" from "sensor" ###
  jsonSensorAngle = jsonSensorData[-1]
  #print(jsonSensor)

  ### get "heading" from "sensor" ###
  jsonSensorHeading = jsonSensorData[0]
  # print(jsonSensor)

  ### get the data ###
  listSensorAngle = jsonSensorAngle["data"]
  listSensorHeading = jsonSensorHeading["data"]
  #print(jsonData)

  ### collect 1000 rows and export csv ###
#  df = pd.DataFrame([])
#  for rowAngle in listSensorAngle:
#      dfRowAngle = pd.DataFrame([rowAngle.split(",")], columns=('ts_a','angle'))
#      # dfRowAngle = rowAngle.split(",")
#      df = df.append(dfRowAngle, ignore_index=True)
#
#  for rowHeading in listSensorHeading:
#      dfRowHeading = pd.DataFrame([rowHeading.split(",")], columns=('ts_b','heading'))
#      # dfRowHeading = rowHeading.split(",")
#      df = dfRowHeading.join(df, lsuffix='_caller', rsuffix='_other')
#
#  print(df)
#  df.to_csv('sensor.csv', header=False, index=False, mode='a', columns=('ts_a','angle','ts_b','heading'))

  # latestAngle = listSensorAngle[-1]
  #
  # if latestAngle == oldAngle:
  #   oldAngle = latestAngle
  # else:
  #   latestAngle = oldAngle
  #   dfRowAngle = pd.DataFrame([latestAngle.split(",")], columns=('ts_a', 'angle'))
  #   dfRowAngle.to_csv('angle.csv', header=False, index=False, mode='a', columns=('ts_a', 'angle'))

  # for rowAngle in listSensorAngle:
  #     dfRowAngle = pd.DataFrame([rowAngle.split(",")], columns=('ts_a', 'angle'))
  #     dfRowAngle.to_csv('angle.csv', header=False, index=False, mode='a', columns=('ts_a', 'angle'))

#   listSensorHeading.sort()
#   oldHeading.sort()
#   if(listSensorHeading == oldHeading):
#
#   for rowHeading in listSensorHeading:
#       dfRowHeading = pd.DataFrame([rowHeading.split(",")], columns=('ts_b', 'heading'))
#       dfRowHeading.to_csv('heading.csv', header=False, index=False, mode='a', columns=('ts_b', 'heading'))

  ### get last element of "data" ###
  AngleData = listSensorAngle[-1]
  HeadingData = listSensorHeading[-1]
  #print(latestData)

  # split with "," to get the value of the sensor data
  SensorAngle = str(AngleData).split(",")[-1]
  SensorHeading = str(HeadingData).split(",")[-1]
  #print(latestDataContent)

  return str(DOName), int(SensorAngle), int(SensorHeading)

### state machine ###
### find average angle difference ###
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

def publishSimulationResult(client, DOName, result_angle):
  current_time = time.time() * 1000
  payload = "{\r\n  \"DO\": \"%s\",\r\n  \"name\":\"pos_3sec_later\",\r\n  \"simargs\":[\"1627475400000, 320\", \"1627475400000, 28\"],\r\n  \"result\" : %d,\r\n  \"ts\": %04d\r\n}\r\n" % (DOName, result_angle, current_time)
  client.publish("dp_sim_data", payload)


def on_message(client, userdata, msg):  # 서버에게서 publish 메세지를 받을 때
  payload = msg.payload.decode('UTF-8')
  jsonPayload = json.loads(payload)
  DOName, boom_angle, heading = ExtractSensorData(jsonPayload)
  publishSimulationResult(client, DOName, simulateSensorData(boom_angle))

def on_connect(client, userdata, flags, rc):
  if rc == 0:
    print("connected")
    client.subscribe(topic)
    client.on_message = on_message
  else:
    print("Bad connection Returned code=", rc)

def on_disconnect(client, userdata, flags, rc=0):
  print("disconnected ......")
  print(str(rc))

if __name__ == "__main__":

    print("Hello...")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect(MQTT_BROKER, 1883)
    client.loop_forever()
    if KeyboardInterrupt == True:
      quit()