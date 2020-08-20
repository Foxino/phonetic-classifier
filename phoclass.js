let mic, fft;

function setup() {
	createCanvas(710, 400);
	noFill();

	mic = new p5.AudioIn();
	mic.start();
	fft = new p5.FFT();
	fft.setInput(mic);
}

function draw() {
	background(200);
	let spectrum = fft.analyze();

	beginShape();
	for (i = 0; i < spectrum.length; i++) {
		vertex(i, map(spectrum[i], 0, 255, height, 0));
	}
	endShape();
}

function saveP() {
	//pressing save should activate this function,
	//saves the last x frames of the fft.analyze to a json for the creation of a NN to classify phonetics
}

function classify() {
	// classifies the last x frames against an NN
}
