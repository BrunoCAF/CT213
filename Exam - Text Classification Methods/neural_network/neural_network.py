from keras import models, activations, layers, losses, optimizers
import utils

def make_neural_network(output_number):
    model = models.Sequential()

    model.add(layers.Dense(20, activation=activations.relu, input_shape=(utils.INPUT_SIZE,)))
    model.add(layers.Dense(20, activation=activations.relu))
    model.add(layers.Dense(output_number, activation=activations.softmax))

    model.compile(loss=losses.mse, optimizer=optimizers.Adam())
    model.summary()

    return model

if __name__ == "__main__":
    model = make_neural_network(5)