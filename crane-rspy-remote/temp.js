var con = 24;
		
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
        console.log(tempToString);