let dataset;
let sample;
let model;
let inputSrc, encoderSrc, decoderSrc;
var representation = [];
let sliders = [];

$('input[type="range"]').rangeslider({
    polyfill : false,
    rangeClass: 'rangeslider',
    disabledClass: 'rangeslider--disabled',
    horizontalClass: 'rangeslider--horizontal',
    verticalClass: 'rangeslider--vertical',
    fillClass: 'rangeslider__fill',
    handleClass: 'rangeslider__handle',
    onInit : function() {
        this.output = $( '<div class="range-output" />' ).insertBefore( this.$range ).html(this.$element.val() );
		representation.push(parseFloat(this.$element.val()));
		sliders.push(this);
    },
    onSlide : function( position, value ) {
        this.output.html( value );
   		representation[parseInt(this.$element[0].id)] = parseFloat(this.$element.val());
		decode(getRep());
 },
	onSlideEnd : function (position, value){
	}
});

$('#reset').click(function () {
	autoencode();
});

$('.config').change(function () {
	setConfig();
	init();
	autoencode();
});

function setConfig(){
	inputSrc = 'img/DATASET/SAMPLE.jpg'.replace('SAMPLE', $('#sampleSelect').val()).replace('DATASET', $('#dataSelect').val());

	encoderSrc = 'models/MODEL/DATASET/encoder/model.json'.replace('MODEL', $('#modelSelect').val()).replace('DATASET', $('#dataSelect').val());
	decoderSrc = 'models/MODEL/DATASET/decoder/model.json'.replace('MODEL', $('#modelSelect').val()).replace('DATASET', $('#dataSelect').val());
};

function getRep(){
	return new Promise( resolve => {
		rep = tf.expandDims(tf.tensor(representation), 0);
		resolve(rep);
	});
}

function setRep(rep){
	representation = rep.dataSync();
	for (i =0 ; i< representation.length; i++){
		sliders[i].$element.val(representation[i]).change();
	}
}
