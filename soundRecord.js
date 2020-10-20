let mic, fft, cnn;
const mouthShapes = ["Neutral", "A1", "A2", "O1", "O2"];

function setup(){
    createCanvas(710, 400);
	noFill();

	mic = new p5.AudioIn();
	mic.start();
	fft = new p5.FFT();
	fft.setInput(mic);
}

function draw() {
	let spectrum = fft.analyze();
	background(255);
	strokeWeight(5);
	let c = color('hsl(160, 100%, 50%)');
	classify(spectrum)
	stroke(c);
	beginShape();
	for (i = 0; i < spectrum.length; i++) {
		vertex(i, map(spectrum[i], 0, 255, height, 0));
	}
	endShape();
}


const allMax = (xs) => xs.reduce(
	(a, x, i) => x > xs[a[0]] ? [i] : x < xs[a[0]] ? [...a] : [...a, i],
	[0]                                
 )
  

function classify(audio){
	if (complete) {
		let x = cnn.predict(tf.tensor(audio,[1,32,32,1]).div(255)).dataSync();
		document.querySelector('#Result').innerHTML = mouthShapes[allMax(x)];

	}
}

function touchStarted() {
	if (getAudioContext().state !== 'running') {
		getAudioContext().resume();
		console.log('Audio Resumed!');
	}
}