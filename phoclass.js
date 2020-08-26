let mic, fft, train, label;
const classes = [ 'aah', 'eee', 'i' ];

const inputSize = 1024; //defaults to 1024, needs to be power of 2.
let sinputShape = Math.sqrt(inputSize);
let numClasses = classes.length;

const epochs = 40;
const learningRate = 0.2;
const testRatio = 0.2;

function setup() {
	train = [];
	label = [];
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
	train.push(spect);
	label.push(cl);
}

function train_nn() {
	// create a conv neural network
	const m = tf.sequential({
		layers: [
			tf.layers.conv2d({
				inputShape: [ sinputShape, sinputShape, 1 ],
				activation: 'relu',
				kernelSize: [ 3, 3 ],
				filters: 64
			}),
			tf.layers.conv2d({
				filters: 64,
				activation: 'relu',
				kernelSize: [ 3, 3 ]
			}),
			tf.layers.maxPooling2d({
				poolSize: [ 2, 2 ],
				strides: [ 2, 2 ]
			}),
			tf.layers.conv2d({
				filters: 64,
				activation: 'relu',
				kernelSize: [ 3, 3 ]
			}),
			tf.layers.conv2d({
				filters: 64,
				activation: 'relu',
				kernelSize: [ 3, 3 ]
			}),
			tf.layers.maxPooling2d({
				poolSize: [ 2, 2 ],
				strides: [ 2, 2 ]
			}),
			tf.layers.conv2d({
				filters: 128,
				activation: 'relu',
				kernelSize: [ 3, 3 ]
			}),
			tf.layers.conv2d({
				filters: 128,
				activation: 'relu',
				kernelSize: [ 3, 3 ]
			}),
			tf.layers.maxPooling2d({
				poolSize: [ 2, 2 ],
				strides: [ 2, 2 ]
			}),
			tf.layers.flatten(),
			tf.layers.dense({
				units: 1024,
				activation: 'relu'
			}),
			tf.layers.dense({
				units: 1024,
				activation: 'relu'
			}),
			tf.layers.dense({
				units: numClasses,
				activation: 'softmax'
			})
		]
	});
	//m.summary();

	const optimizer = 'rmsprop';

	m.compile({
		optimizer,
		loss: 'categoricalCrossentropy',
		metrics: [ 'accuracy' ]
	});

	for (let x = 0; x < train.length; x++) {
		if (Math.random() < testRatio) {
			// valdiation data
		} else {
			// train data
		}
	}

	// load data into tensors

	//for (let x = 0; x < train.length; x++) {}

	// split into training and testing data.

	//fit

	//save
}

train_nn();

function classify() {
	// every x frames classify the current phonetic input against the CNN.
	return 'N/A';
}
