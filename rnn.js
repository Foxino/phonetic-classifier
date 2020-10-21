

async function rnnModel(){

    // Load in DataSet
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
    
    // TO DO - split dataset into test and train


    //prepare dataset
    const xs = tf.tensor(dataset.x).div(255);
    const ys = tf.tensor(processed_y);

    const model = tf.sequential();

    model.add(
        tf.layers.dense({
            inputShape: [1024],
            units: 128
        })
    )


    //Build RNN




    model.add(tf.layers.reshape({
        targetShape: [128, 5]
    }))

    let lstm_cells = [];
    for (let index = 0; index < 5; index++) {
         lstm_cells.push(tf.layers.lstmCell({units: 128}));
    }

    model.add(tf.layers.rnn({
        cell: lstm_cells,
        inputShape: [128,5],
        returnSequences: false
    }));


    model.add(tf.layers.dense({
        inputShape: [5],
        units: 128,
        activation: 'relu'
    }))

    model.add(tf.layers.dropout(.2))
    
    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu'
    }))

    model.add(tf.layers.dropout(.2))

    model.add(tf.layers.dense({
        units: 64,
        activation: 'relu'
    }))

    model.add(tf.layers.dropout(.2))

    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu'
    }))

    model.add(tf.layers.dropout(.2))

    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu'
    }))

    model.add(tf.layers.dense({
        units: 5,
        activation: 'softmax'
    }))

    model.compile({
        optimizer: tf.train.adam(.01),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const h = await model.fit(xs, ys, {
        batchSize: 100,
        epochs: 20,
        callbacks:{
            onEpochEnd: async(epoch, log) =>{
                callback(epoch,log);
            }
        }
    });

    return model
}