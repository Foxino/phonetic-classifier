let mic, fft, train, label, phocnn;

let training_x = [],
	training_y = [],
	testing_x = [],
	testing_y = [];

const classes = [ 'aah', 'eee', 'i' ];

const inputSize = 1024; //defaults to 1024, needs to be power of 2.
let sinputShape = Math.sqrt(inputSize);
let numClasses = classes.length;

// Training Paras
const epochs = 400;
const learningRate = 0.2;
const testRatio = 0.2;
const batchSize = 40;

// Classification Paras
const predictionConfidence = 0.7;
const predictionInterval = 0;

//stuff
let train_t_x, train_t_y, test_t_x, test_t_y;

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
	let arr = [];
	let cl = document.getElementById('ph').selectedIndex;
	for (let x = 0; x < numClasses; x++) {
		if (cl == x) {
			arr.push(1);
		} else {
			arr.push(0);
		}
	}
	var spect = fft.analyze();
	train.push(spect);
	label.push(arr);
}

function createCNN() {
	// creates a convoluted neural network
	const model = tf.sequential();

	//input layer
	model.add(
		tf.layers.conv2d({
			inputShape: [ sinputShape, sinputShape, 1 ],
			kernelSize: 3,
			filters: 16,
			activation: 'relu'
		})
	);

	model.add(
		tf.layers.maxPooling2d({
			poolSize: 2,
			strides: 2
		})
	);

	model.add(
		tf.layers.conv2d({
			kernelSize: 3,
			filters: 32,
			activation: 'relu'
		})
	);

	model.add(
		tf.layers.maxPooling2d({
			poolSize: 2,
			strides: 2
		})
	);

	model.add(
		tf.layers.conv2d({
			kernelSize: 3,
			filters: 32,
			activation: 'relu'
		})
	);

	model.add(tf.layers.flatten({}));

	model.add(
		tf.layers.dense({
			units: 64,
			activation: 'relu'
		})
	);

	model.add(
		tf.layers.dense({
			units: numClasses,
			activation: 'softmax'
		})
	);

	return model;
}

function loadData() {
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
	//load into tensors
	train_t_x = tf.tensor(training_x, [ training_x.length, 32, 32, 1 ]);
	train_t_y = tf.tensor(training_y);

	test_t_x = tf.tensor(testing_x, [ testing_x.length, 32, 32, 1 ]);
	test_t_y = tf.tensor(testing_y);
}

function train_nn() {
	console.log('Creating Model.. ');
	let cnn = createCNN();
	console.log('Loading Data into Tensors.. ');
	loadData();

	const optimizer = 'rmsprop';

	cnn.compile({
		optimizer,
		loss: 'categoricalCrossentropy',
		metrics: [ 'accuracy' ]
	});

	fit(train_t_x, train_t_y, test_t_x, test_t_y, cnn).then(() => {
		let t = cnn.predict(test_t_x);
		console.log('Test Data');
		t.print();
		test_t_y.print();
		phocnn = cnn;
		console.log('We good homie?');
	});

	// save model to disk once done
}

async function fit(train_x, train_y, test_x, test_y, model) {
	const response = await model.fit(train_x, train_y, {
		batchSize,
		testRatio,
		epochs: epochs,
		callbacks: {
			onBatchEnd: async (batch, logs) => {
				console.log(logs);
				await tf.nextFrame();
			},
			onEpochEnd: async (epoch, logs) => {
				console.log(logs);
				await tf.nextFrame();
			}
		}
	});
}

function classify() {
	if (typeof phocnn !== 'undefined') {
		//return prediction based on highest confidence
		//let p = phocnn.predict(tf.tensor(fft.analyze(), [ 1, 32, 32, 1 ]));
		//clear tensors from gpu
	} else {
		return 'N/A';
	}
}
