function inputImage(src){
    return new Promise(resolve => {
		img = document.createElement('img');
		canvas = document.getElementById('inputCanvas');
		ctx = canvas.getContext("2d");

        img.onload = function () {
			imgTensor = tf.browser.fromPixels(this, numChannels=3); 
			imgTensor = tf.div(imgTensor, 255);
			ctx.drawImage(img,0,0, 200,200);
            resolve(tf.expandDims(imgTensor, 0));
       	}
    	img.src = src;
    });
};

function normalize(image){
	image = tf.minimum(image,1)
	image = tf.maximum(image,0)
	return image;
};

function outputImage(img){
	let outputCanvas = document.getElementById('outputCanvas');
	normImg = normalize(tf.squeeze(img));
	
	scaledImg = tf.image.resizeBilinear(normImg, [outputCanvas.width, outputCanvas.height]);
	tf.browser.toPixels(scaledImg, outputCanvas, 200,200); 
};

async function loadEncoder(src) {
	return await tf.loadGraphModel(src);
};
async function loadDecoder(src) {
	return await tf.loadGraphModel(src);
};

function encode(){
	return Promise.all([inputImg, encoder]).then( (values) => {
		img = values[0];
		encoder = values[1];
		rep = encoder.predict(img)[0];
		setRep(rep);
		return (rep);
	});
};

function decode(rep){
	Promise.all([decoder, rep]).then( (values) => {
		decoder = values[0];
		rep = values[1];
		let output = decoder.predict(rep)[1];
		outputImage(output);
	}
	);
};

function autoencode(){
	rep = encode();
	decode(rep);
}
let inputImg;
let encoder;
let decoder;

function init(){
	inputImg = inputImage(inputSrc);
	encoder = loadEncoder(encoderSrc);
	decoder = loadDecoder(decoderSrc);
};

$('document').ready(function (){
	setConfig();
	init();
	autoencode();
});
