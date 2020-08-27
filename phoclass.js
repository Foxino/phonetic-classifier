let mic, fft, train, label;

let training_x = [],
	training_y = [],
	testing_x = [],
	testing_y = [];

const classes = [ 'aah', 'eee', 'i' ];

const inputSize = 1024; //defaults to 1024, needs to be power of 2.
let sinputShape = Math.sqrt(inputSize);
let numClasses = classes.length;

// Training Paras
const epochs = 40;
const learningRate = 0.2;
const testRatio = 0.2;
const batchSize = 10;

// Classification Paras
const predictionConfidence = 0.7;
const predictionInterval = 0;

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

function touchStarted() {
	if (getAudioContext().state !== 'running') {
		getAudioContext().resume();
		console.log('Audio Resumed!');
	}
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
			tf.layers.conv1d({
				inputShape: [ 1, sinputShape, sinputShape ],
				activation: 'relu',
				kernelSize: 3,
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

	m.summary();

	// x = data, y = label

	for (let x = 0; x < train.length; x++) {
		// randomly split the test/training data
		if (Math.random() < testRatio) {
			// valdiation data
			testing_x.push(train[x]);
			testing_y.push(label[x]);
		} else {
			// train data
			training_x.push(train[x]);
			training_y.push(label[x]);
		}
	}

	console.log('Fitting Data.. ');

	const train_t_x = tf.tensor(training_x, [ training_x.length, 32, 32 ]);
	const train_t_y = tf.tensor(training_y);

	const test_t_x = tf.tensor(testing_x, [ testing_x.length, 32, 32 ]);
	const test_t_y = tf.tensor(testing_y);

	//fit

	fit(train_t_x, train_t_y, test_t_x, test_t_y, m).then(() => {
		console.log('Complete');
	});

	//save
}

async function fit(train_x, train_y, test_x, test_y, model) {
	const response = await model.fit(train_x, train_y);
	console.log(response);
}

function classify() {
	// every x frames classify the current phonetic input against the CNN.
	return 'N/A';
}
