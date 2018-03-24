## TensorFlow template
We have 3 `.py` files and 2 folders which are described below.

* `main.py`
    * Define the whole process.

* `model.py`
    * Define the architecture of the model.
    * methods:
        - build: build the model.
        - sub_model: define the architecture of the model.
        - predict: predict the labels according to the model.
        - loss: define the loss of the model.

* `trainer.py`
    * Define how to train the model.
    * methods:
        - train: train the model.
        - result: show the result. 
    * arguments:
        - sess: tf.sess declared outside Trainer
        - epoch: the number of epoch we need to train
        - print_every: how often we print the result

* logs
    * Record the training process
    * Usage: python -m tensorboard.main --logdir=path/to/logs

* result
    * The visualized result.
