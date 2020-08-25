let mic, fft, train;
const classes = [ 'aah', 'eee', 'i' ];

const inputSize = 1024; //defaults to 1024, needs to be power of 2.
let inputShape = Math.sqrt(inputSize);
let numClasses = classes.length;

class trainingData {
	constructor(clas, spect) {
		this.class = clas;
		this.spectrum = spect;
	}
}

function setup() {
	train = [];
	createCanvas(710, 400);
	noFill();

	mic = new p5.AudioIn();
	mic.start();
	fft = new p5.FFT();
	fft.setInput(mic);
}

function draw() {
	background(255);
	let spectrum = fft.analyze();
	strokeWeight(5);
	let c = color('hsl(160, 100%, 50%)');
	stroke(c);

	document.getElementById('result').innerHTML = classify(spectrum);

	beginShape();
	for (i = 0; i < spectrum.length; i++) {
		vertex(i, map(spectrum[i], 0, 255, height, 0));
	}
	endShape();
}

function saveP() {
	var cl = document.getElementById('ph').selectedIndex;
	var spect = fft.analyze();
	train.push(new trainingData(cl, spect));
}

function train_nn() {
	// create a conv neural network
	const m = tf.sequential(
		[
			//layers
		]
	);

	//transform training data into tensors

	//fit

	//save
}

function classify() {
	// every x frames classify the current phonetic input against the CNN.
	return 'N/A';
}
