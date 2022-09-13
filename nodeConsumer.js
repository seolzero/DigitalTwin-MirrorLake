const Influx = require('influx');
const influx = new Influx.InfluxDB('http://localhost:8086/mirrorlake')
const kafka = require('kafka-node'),
    Consumer = kafka.Consumer,
    client = new kafka.KafkaClient({kafkaHost: "172.26.240.22:9092"}),
    consumer = new Consumer(
        client,
        [
            { topic: 'DO_DOcrn001', partition: 0 }
        ],
        {
            autoCommit: false
        }
    );

let jObject = {}, parseObject;
consumer.on('message', function (message) {
   //console.log(message);
   parseObject = JSON.parse(message.value)
   jObject = {
      "angle" : parseObject.sensor1_value,
      "heading" : parseObject.sensor2_value,
      "timestamp": parseObject.sensor1_rowtime
   };
   console.log(jObject)
   writeToInflux(jObject)

});

//{"tmpA":1,"sensor1_rowtime":"2022-08-25 16:53:10.961","sensor1_id":"angle","sensor1_value":"38","sensor2_id":"heading","sensor2_value":"60"}
function writeToInflux(ContentsData){		
	influx.writePoints([
		{ measurement: 'crain',
			fields: ContentsData, 
			// timestamp: 1641970000000000000
			}
	 ],{
		// precision: 'ns',
	}).then(result => {
      console.log(">>> Send Response (Influx), 200");
   }).catch(err => {
		console.error(`Error saving data to InfluxDB! ${err.stack}`)
	});

}
// writeToInflux();
