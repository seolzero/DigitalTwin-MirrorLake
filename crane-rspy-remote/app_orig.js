
const express = require('express');
const app = express();
const mqtt = require('mqtt');

const port = 7576;
app.listen(port, () => console.log(`Example app listening at http://localhost:${port}`))

var dt = new Date();
console.log("date: ", dt);

const options = {
  host: 'localhost',
  port: 1883,
  protocol: 'mqtts'
};

const client = mqtt.connect('mqtt://localhost:1883');

client.on("connect", () => {	
  console.log("connected "+ client.connected);
});


client.on("error", (error) => {
  console.log("Can't connect" + error);
});


var topic, con, pubData;
const topicList = ['/oneM2M/req/*/Crane/*', '/oneM2M/req/*/controlButton_sub/*'];

app.post("*", (req, res)=>{
	var Url = req.originalUrl;
	console.log(">> Url:", Url); //  /bada/Crane_02/angle 맨앞에 공백 있음
	var fullBody = '';

	req.on('data', function(chunk) {
		fullBody += chunk; 
	});

	req.on('end', function() {
		res.status(200).send('post /end test ok');
		console.log("------------ fullBody ----------")
		console.log(fullBody);
		console.log("-----------------------------------")
		var receivedData = JSON.parse(fullBody); // { 'm2m:cin': { con: 32 } }

		con = receivedData['m2m:cin'].con;
		console.log("con: ", con); // 32
		var conTempToString = JSON.stringify(con);
		console.log("[url]:", Url);
		console.log("[fullBody]: ", fullBody);

		var path = Url.split('/');
		var ae = path[2];
      	var container = path[3];
      	//console.log(path[0],ae,container, "con: ", conTempToString);
		/*
		console.log("path[0]: ",path[0]); // 공백
		console.log("path[1]: ",path[1]); // bada
		console.log("path[2]: ",path[2]); // Crane_01
		console.log("path[3]: ",path[3]); // angle
		*/
		console.log(">> ae: ", ae);
		console.log(">> container: ", container);
		
		pubData = `{
			"op": 5,
			"net": "3",
			"fr": "/Mobius",
			"rqi": "HJ4gRjBw0b",
			"pc": {
			  "m2m:sgn": {
				"net": "3",
				"sur": "Mobius/${ae}/${container}/sub",
				"nec": "",
				"nev": {
				  "rep": {
					"m2m:cin": {
					  "rn": "4-20171101132736315EaAz",
					  "ty": 4,
					  "pi": "HkzlnObwAb",
					  "ri": "B1Ml0irD0W",
					  "ct": "20211101T132736",
					  "et": "20211101T132736",
					  "lt": "20211101T132736",
					  "st": 3,
					  "cs": 2,
					  "con": ${conTempToString},
					  "cr": "S20170717074825768bp2l"
					}
				  }
				}
			  }
			}
		  }`;


		  var temp = JSON.parse(pubData);
		  var tempToString = JSON.stringify(temp);
		  //console.log("tempToString", tempToString);
		  console.log("[stringified pubData]: ", tempToString);
		  console.log();
//					  "con": ${con},
//"con": {\"concurrent_command\": [ \"boom_up\" ] },

      	//console.log("container:" + container);
		if(container == 'distance' || container == 'heading' || container == 'angle'){
			topic = topicList[0];
		}else if(container == 'controlButton'){
			topic = topicList[1];
		}

		//topic = "/oneM2M/req/*/controlButton_sub/*";
		          
		client.publish(topic, tempToString, options);
		
	/*	
		client.publish(topic, pubData, (error) => {

			console.log("pub error_" + error);
		});
	*/
	});

})

