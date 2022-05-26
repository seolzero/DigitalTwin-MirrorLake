// for GY-85
#include <Wire.h>
#include "GY_85.h"
#include <HMC5883L.h>

// I2Cdev and MPU9150 must be installed as libraries, or else the .cpp/.h files
// for both classes must be in the include path of your project
#include "I2Cdev.h"
#include "MPU9150.h"
#include "helper_3dmath.h"

// for send data to bada(ethernet)
#include <ArduinoJson.h>
#include <SPI.h>
#include <Ethernet.h>
#include <Adafruit_SleepyDog.h>

//create the object
GY_85 GY85;     
HMC5883L compass;

// class default I2C address is 0x68
// specific I2C addresses may be passed as a parameter here
// AD0 low = 0x68 (default for InvenSense evaluation board)
// AD0 high = 0x69
MPU9150 accelGyroMag(0x69);

// ====for compass & angle variable
int16_t ax, ay, az;
int16_t gx, gy, gz;
int16_t mx, my, mz;

// for angle
boolean firstCheck = true;
boolean secondCheck = true; 

double pitch = 0;
double roll = 0;

int16_t roll_current, roll_old, roll_older;
int16_t roll_sum, roll_avg;

// for heading
boolean headingNoiseCheck = false;
int16_t heading_current = -1;
int16_t heading_old = 0; int16_t heading_older = 0;

float heading;
float headingDegrees;

int RoundHeadingDegreeInt;
int RoundAngleDegreeInt;

int PreviousHeadingDegree = 0;
int PreviousAngleDegree = 0;

int error = 0;

// ====for send data to bada(ethernet) variable
// Enter a MAC address for your controller below.
// Newer Ethernet shields have a MAC address printed on a sticker on the shield


//IPAddress httpPostServerIP(203,254,173,126);
IPAddress httpPostServerIP(192, 168, 10, 9);



//SETUP ********

//Crane1(RED), Heading/Angle Module
byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };
//Crane2(YELLOW), Heading/Angle Module
//byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xEC };

//Bada SENSORS
//String aeNameForBada = "Crane_01";
String aeNameForBada = "Crane_02";


String angleContainerForBada = "angle";
String headingContainerForBada = "heading";


String headerForBada = "Accept:application/json, text/plain, */*\nContent-Type:application/json;"; 
int responseCode = 0;

// Initialize the Ethernet client library
// with the IP address and port of the server
// that you want to connect to (port 80 is default for HTTP):
EthernetClient client;

void setup()
{
    // ============================ to connect gy-85 & mpu9150
    Wire.begin();
    delay(10);
    Serial.begin(9600);
    delay(2000);
    GY85.init();
    delay(10);

    compass = HMC5883L();
    
    accelGyroMag.initialize();

    Serial.println("Setting scale to +/- 1.3 Ga");
    error = compass.SetScale(1.3); // Set the scale of the compass.
    if(error != 0) // If there is an error, print it out.
      Serial.println(compass.GetErrorText(error));
  
    Serial.println("Setting measurement mode to continous.");
    error = compass.SetMeasurementMode(Measurement_Continuous); // Set the measurement mode to Continuous
    
    if(error != 0) // If there is an error, print it out.
      Serial.println(compass.GetErrorText(error)); 

    // ============================ to connect ethernet
    // You can use Ethernet.init(pin) to configure the CS pin
    Ethernet.init(10);  // Most Arduino shields

    // start the Ethernet connection:
    Serial.println("Initialize Ethernet with DHCP:");
    
    if (Ethernet.begin(mac) == 0) {
      Serial.println("Failed to configure Ethernet using DHCP");
    
      // Check for Ethernet hardware present
      if (Ethernet.hardwareStatus() == EthernetNoHardware) {
        Serial.println("Ethernet shield was not found.  Sorry, can't run without hardware. :(");
      }
      if (Ethernet.linkStatus() == LinkOFF) {
        Serial.println("Ethernet cable is not connected.");
      }
      // try to congifure using IP address instead of DHCP:
      //Ethernet.begin(mac, ip, myDns);
    } else {
      Serial.print("  DHCP assigned IP ");
      Serial.println(Ethernet.localIP());
    }
    
    // give the Ethernet shield a second to initialize:
    delay(1000);
    
}


void loop()
{
    // these methods (and a few others) are also available
    //accelGyroMag.getAcceleration(&ax, &ay, &az);
    accelGyroMag.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    
    // Retrive the raw values from the compass (not scaled).
    MagnetometerRaw raw = compass.ReadRawAxis();
    // Retrived the scaled values from the compass (scaled to the configured scale).
    MagnetometerScaled scaled = compass.ReadScaledAxis();

    mx = GY85.compass_x( GY85.readFromCompass() );
    my = GY85.compass_y( GY85.readFromCompass() );
    mz = GY85.compass_z( GY85.readFromCompass() );
    
    //Serial.print(">>>>"); Serial.println(raw.XAxis);
    
    getHeading();
    getAngle();
    //headingDataNoiseCheck();

    //if(headingNoiseCheck)
    //{
      //RoundHeadingDegreeInt = heading_older;
    //}
    
    /*Serial.print  ( "1. accelerometer" );Serial.println();
    Serial.print  ( "ax:" );Serial.print( ax ); Serial.print( " ay:" ); Serial.print( ay );Serial.print(" az:");Serial.print( az );Serial.println();
    Serial.print  ( "roll:" );Serial.print( roll ); Serial.print  ( " pitch:" );Serial.print( pitch );Serial.println();Serial.println();
    Serial.print  ( "roll_current:" );Serial.print( roll_current ); Serial.print(" roll_old:" );Serial.print( roll_old );Serial.print(" roll_older:" );Serial.print(roll_older);Serial.println();Serial.println();
    
    Serial.print  ( "2. compass" );Serial.println();
    Serial.print  ( "mx:" );Serial.print( mx ); Serial.print(" my:" ); Serial.print( my );Serial.print(" mz:");Serial.print( mz );Serial.println();
    Serial.print  (" heading: ");Serial.print(RoundHeadingDegreeInt); Serial.println();Serial.println();
    //Serial.print  ( "heading_current:" );Serial.print( heading_current ); Serial.print(" heading_old:" );Serial.print( heading_old );Serial.print(" heading_older:" );Serial.print( heading_older );Serial.println();Serial.println();
    */

    Serial.println("----------");
    String jsonBodyForBadaAngleSensor = makeJsonForBada(angleContainerForBada);
    //Serial.println("3. send angle value to bada ");Serial.println(jsonBodyForBadaAngleSensor);
    
    responseCode = httpPostRequest(httpPostServerIP, 1220, "/DigitalConnector/SensorGroup/" + angleContainerForBada, headerForBada, jsonBodyForBadaAngleSensor);
//    responseCode = httpPostRequest(httpPostServerIP, 8080, "/DigitalConnector/SensorGroup/" + angleContainerForBada, headerForBada, jsonBodyForBadaAngleSensor);
    
    //Serial.print("angle responsecode: ");Serial.print(responseCode); Serial.println();
    Serial.print(">> sent 1, angle  ");
    delay(500);
    
    String jsonBodyForBadaHeadingSensor = makeJsonForBada(headingContainerForBada);
    //Serial.println("4. send heading value to bada"); Serial.println(jsonBodyForBadaHeadingSensor);
    responseCode = httpPostRequest(httpPostServerIP, 1220, "/DigitalConnector/SensorGroup/" + headingContainerForBada, headerForBada, jsonBodyForBadaHeadingSensor);
//    responseCode = httpPostRequest(httpPostServerIP, 8080, "/DigitalConnector/SensorGroup/" + headingContainerForBada, headerForBada, jsonBodyForBadaHeadingSensor);
    
    //Serial.print("heading responsecode: ");Serial.print(responseCode); Serial.println();
    Serial.println(">> sent 2, heading");
    delay(900);

    //Serial.println("==============");
    
       
}

void getHeading(void)
{
  //heading = 180 * atan2(my, mx) / PI;
  heading = atan2(my, mx);

 /*
  * //magnetic declination: http://www.magnetic-declination.com/
 * 대한민국 서울: -8º4' W = -8.067º W = -0.1408 radian
 */
  float declinationAngle = - 0.1408;
  heading += declinationAngle;

  // Correct for when signs are reversed.
  if( heading < 0) heading += 2*PI;

  // Check for wrap due to addition of declination.
  if( heading > 2*PI) heading -= 2*PI;

  // Convert radians to degrees for readability.
  headingDegrees = heading * 180/M_PI;

  //rounding the heading
  RoundHeadingDegreeInt =round(headingDegrees);
  Serial.println(RoundHeadingDegreeInt);
  
  //smoothing value
  //if( RoundHeadingDegreeInt < (PreviousHeadingDegree + 3) && RoundHeadingDegreeInt > (PreviousHeadingDegree - 3) ) {
    //RoundHeadingDegreeInt = PreviousHeadingDegree;
 //}

  //PreviousHeadingDegree = RoundHeadingDegreeInt;
}

void getAngle(void)
{
  
  pitch = atan2(-ax, sqrt(ay*ay + az*az)) * 180.0/M_PI;
  roll  = atan2(ay, az) * 180.0/M_PI;

  //first
  if(firstCheck)
  {
    firstCheck = false;
    roll_current = roll;
    roll_avg = roll_current;
  }
  else
  {
    if(secondCheck)
    {
      secondCheck = false;
      roll_old = roll_current;
      roll_current = roll;
      
      roll_sum = roll_old + roll_current;
      roll_avg = roll_sum/2;
    }
    else
    {
      roll_older = roll_old;
      roll_old = roll_current;
      roll_current = roll;
      
      roll_sum = roll_older + roll_old + roll_current;
      roll_avg = roll_sum/3;
    }
  }
  
  RoundAngleDegreeInt =round(roll_avg);
  
  //smoothing value
  //if( RoundAngleDegreeInt < (PreviousAngleDegree + 7) && RoundAngleDegreeInt > (PreviousAngleDegree - 7) ) {
    //RoundAngleDegreeInt = PreviousAngleDegree;
  //} 

  //PreviousAngleDegree = RoundAngleDegreeInt;
}

void headingDataNoiseCheck(void)
{
  // first
  if(heading_current == -1)
  {
    heading_current = RoundHeadingDegreeInt;
    heading_old = heading_current;
  }
  else
  {
    heading_current = RoundHeadingDegreeInt;
  }

  // noise check
  if(abs(heading_current - heading_old) < 3)
  {
    heading_old = heading_current;
    headingNoiseCheck = false;
  }
  else
  {
    if(headingNoiseCheck)
    {
      heading_old = heading_current;
      headingNoiseCheck = false;
    }
    else
    {
      heading_older = heading_old;
      heading_old = heading_current;
      headingNoiseCheck = true;
    }
  }
}

String makeJsonForBada(String value)
{
//  StaticJsonBuffer<400> jsonBuffer;
//  JsonObject& root = jsonBuffer.createObject();
  StaticJsonDocument<400> jsonBuffer;
  
  JsonObject root = jsonBuffer.to<JsonObject>();
  JsonObject cin = jsonBuffer.createNestedObject("data");

  if (value.equals(angleContainerForBada)) 
  {
    root["data"] = RoundAngleDegreeInt;
  }
  else if (value.equals(headingContainerForBada)) 
  {
    root["data"] = RoundHeadingDegreeInt;
  } 
  else 
  {
    Serial.println("makeJsonForBada failed");
  }
  
  String jsonData;
//  root.printTo(jsonData);
  serializeJson(root, jsonData);
  Serial.print("root: ");
  Serial.println(root);
  Serial.print("jsonData: ");
  Serial.println(jsonData);
  

  return jsonData;
}

int httpPostRequest(IPAddress serverIP, int portNum, String url, String header, String body)
{
  int statusCode = 408;  // http timeout status code
  String readBuffer = "";
  unsigned long setTime = 0;
  unsigned long timeout = 3000L; // should check the unit
  boolean got_response = false;

  /*Serial.print("serverIP: ");
  Serial.println(serverIP);
  Serial.print("portNum: ");
  Serial.println(portNum);
  Serial.print("url: ");
  Serial.println(url);
  Serial.print("header: ");
  Serial.println(header);
  Serial.print("body: ");
  Serial.println(body);*/
    
  if (client.connect(serverIP, portNum)) {
    //Serial.println("connection success");
    
    client.print("POST ");  //"POST /CoT/base HTTP/1.1"
    client.print(url);
    client.println(" HTTP/1.1");

    client.print("Host: ");
    client.print(serverIP);
    client.print(":");
    client.println(portNum);

    client.println(header);
    client.println("User-Agent: Arduino/1.0");
    client.print("Content-Length: ");
    client.println(body.length());
    client.println("Connection: close");
    client.println();
    client.println(body);

    setTime = millis();
    while ((millis() - setTime) < timeout  || got_response != true) 
    {
       int len = client.available();
       if (len > 0) 
       {
          byte read_buffer[1024];
          client.read(read_buffer, len);
          Serial.write(read_buffer, len);
          got_response = true;
          //statusCode = read_buffer.substring(9, 12).toInt();
          statusCode = 201;
          return statusCode;
       }
    }

    statusCode = 408;
    return statusCode;
 
 /*   
    if (readBuffer.length() > 0) {
      Serial.println("========================");
      Serial.print("readBuffer: ");
      Serial.println(readBuffer);
      Serial.println("========================");
      statusCode = readBuffer.substring(9, 12).toInt();
      //break;
    }
    */
    //}
    
  } else {
    statusCode = 404;  // connection fail, 404 not found
  }
  
  client.stop();
  return statusCode;
}
