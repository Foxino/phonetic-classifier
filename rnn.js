

async function rnnModel(){
    
    //prepare dataset
    const xs = tf.tensor(dataset.x).div(255);
    raw_y = []
    for (let x = 0; x < dataset.y.length; x++) {
        raw_y.push([dataset.y[x].m, dataset.y[x].w, dataset.y[x].trL, dataset.y[x].trR]);
    }
    const ys = tf.tensor2d(raw_y);

    //xs.print();
    //ys.print();

    console.log(xs.shape)

    const model = tf.sequential();

    model.add(tf.layers.dense({
        units: 128,
        

    }))

    model.add(
		tf.layers.lstm({
            activation: 'relu',
            returnSequences: true,
			units: 128
		})
    );
    
    model.add(tf.layers.dropout(.2))

    model.add(
		tf.layers.lstm({
            activation: 'relu',
            returnSequences: true,
			units: 128
		})
    );

    model.add(tf.layers.dense({
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
        units: 4
    }))

    model.compile({
        optimizer: tf.train.adam(.01),
        loss: 'meanSquaredError'
    });

    const h = await model.fit(xs, ys, {
        batchSize: 5000,
        epochs: 20,
        callbacks:{
            onEpochEnd: async(epoch, log) =>{
                callback(epoch,log);
            }
        }
    });

    return model
}