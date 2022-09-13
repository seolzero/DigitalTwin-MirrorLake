
const express = require('express');
const app = express();
const mqtt = require('mqtt');
const http = require('http')
var request = require('request')

const port = 7576;

var cnt_angle = 0;
var cnt_heading = 0;

app.listen(port, () => console.log(`Example app listening at http://localhost:${port}`))

var dt = new Date();
console.log("date: ", dt);

const options = {
  //host: 'localhost',
  host: '192.168.1.116',
  port: 1883,
  protocol: 'mqtts'
};

//const client = mqtt.connect('mqtt://localhost:1883');
const client = mqtt.connect('mqtt://192.168.1.116:1883');

client.on("connect", () => {	
  console.log("connected "+ client.connected);
});


client.on("error", (error) => {
  console.log("Can't connect" + error);
});


var topic, con, pubData, unreal_post_data;
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
		/**/
		console.log("------------ fullBody ----------")
		console.log(fullBody);
		console.log("-----------------------------------")
		
		var receivedData = JSON.parse(fullBody); // { 'm2m:cin': { con: 32 } }

		var path = Url.split('/');
		var base = path[1]; // rpi: Bada, unity: bada
		var ae = path[2]; 
      	var container = path[3];

		if(base == 'bada')
		{
			//console.log(">>>>>> bada \n");
			con = receivedData['m2m:cin'].con;
			console.log("con: ", con); // 32
			var conTempToString = JSON.stringify(con);
			console.log("[url]:", Url);
			console.log("[fullBody]: ", fullBody);


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

			unreal_post_data = '';
			
			unreal_post_data = `{
									"m2m:cin":
									{
										"con": ${conTempToString}
									}
								}`;
			unreal_post_data = JSON.parse(unreal_post_data);

			request.post({
				headers: {'content-type':'application/json'},
				uri: `http://localhost:3500/bada/${ae}/${container}`,
				body: unreal_post_data,
				json:true
			},function(error, response,body){
				if(response){
					console.log(response.body);
					//console.log("body: ",response.statusCode, response.body);
				}
			});


			var temp = JSON.parse(pubData);
			var tempToString = JSON.stringify(temp);
			//console.log("tempToString", tempToString);
			console.log("[stringified pubData]: ", tempToString);
			console.log();

			if(container == 'heading'){
				cnt_heading++;
				topic = topicList[0];
			}
			else if(container == 'angle'){
				cnt_angle++;
				topic = topicList[0];
			}
			else if(container == 'controlButton'){
				topic = topicList[1];
			}

			/*
			if(container == 'distance' || container == 'heading' || container == 'angle'){
				topic = topicList[0];
			}else if(container == 'controlButton'){
				topic = topicList[1];
			}
			*/
			//topic = "/oneM2M/req/*/controlButton_sub/*";
					
			client.publish(topic, tempToString, options);
			console.log("topic: ", topic);
			
		} // base == "bada"
		else if(base == 'Bada')
		{
			//console.log(">>>>>> Bada \n");
			if(receivedData['m2m:cin'])
			{
				con = receivedData['m2m:cin'].con;
				//console.log("Bada con: ", con); //  { concurrent_command: [ 'hook_down' ] }
				var conTempToString = JSON.stringify(con);

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

				unreal_post_data = '';
			
				unreal_post_data = `{
										"m2m:cin":
										{
											"con": ${conTempToString}
										}
									}`;
				unreal_post_data = JSON.parse(unreal_post_data);
	
				request.post({
					headers: {'content-type':'application/json'},
					uri: `http://localhost:3500/Bada/${ae}/${container}`,
					body: unreal_post_data,
					json:true
				},function(error, response,body){
					if(response){
						console.log(response.body);
						//console.log("body: ",response.statusCode, response.body);
					}
				});

				var temp = JSON.parse(pubData);
				var tempToString = JSON.stringify(temp);
				//console.log("tempToString", tempToString);
				console.log("       [stringified pubData]: ", tempToString);
				console.log();

				if(container == 'controlButton')
				{
					topic = topicList[1];
					client.publish(topic, tempToString, options);
				}
				else
				{
					console.log("***** container: ", container);
				}

			}


		}
	});

	function print_cnt(){
		var now = new Date();
		console.log(">> now: ", now);
		console.log("================\ncnt_angle = %d",cnt_angle);
		console.log("cnt_heading = %d\n================",cnt_heading);
		cnt_angle = 0;
		cnt_heading = 0;
	}

	let timerId = setInterval(() => print_cnt(), 1000);

})

