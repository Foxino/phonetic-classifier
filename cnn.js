const testRatio = .2;
const batchSize = 100;
const epochs = 50;

async function cnnModel(){
	//tf.engine().startScope()

    testing_x = []
    training_x = []

    testing_y = []
	training_y = []

	processed_y = []
	
	for (let x = 0; x < dataset.y.length; x++) {
		let arr = [0,0,0,0,0]
		for (let y = 0; y < arr.length; y++) {
			if(dataset.y[x] == y){
				arr[y] = 1;
			}
		}
		processed_y.push(arr);
	}

    for (let x = 0; x < dataset.x.length; x++) {
		// randomly split the test/training data
		if (Math.random() < testRatio) {
			// valdiation data
			testing_x.push(dataset.x[x]);
			testing_y.push(processed_y[x]);
		} else {
			// train data
			training_x.push(dataset.x[x]);
			training_y.push(processed_y[x]);
		}
	}


    const xs = tf.tensor(training_x, [ training_x.length, 32, 32, 1 ]).div(255);
    const ys = tf.tensor(training_y);

    const t_xs = tf.tensor(testing_x, [ testing_x.length, 32, 32, 1 ]).div(255);
    const t_ys = tf.tensor(testing_y);

	const model = tf.sequential();
	
	xs.print();
	ys.print();

	//input layer
	model.add(
		tf.layers.conv2d({
			inputShape: [ 32, 32, 1 ],
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
			units: 5,
			activation: 'softmax'
		})
    );
    
    model.compile({
        optimizer: tf.train.adam(.01),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    fit(xs, ys, t_xs, t_ys, model).then(() => {
		console.log('We good homie?');
		cnn = model;
		complete = true;
	});
	//tf.engine().endScope()


	return model;

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