let mic, fft, train;
const classes = [ 'aah', 'eee', 'i' ];

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
	//
}

function classify() {
	return 'N/A';
}
