from utils import process_text_entry, format_labels
from keras import models
from neural_network import make_neural_network

OUTPUT_SIZE = 5
num_epochs = 1000

model = make_neural_network(OUTPUT_SIZE)

training_inputs, batch_size = process_text_entry('Training/Questions.txt')

expected_outputs = format_labels('Training/Tags.txt')

model.fit(training_inputs, expected_outputs, batch_size=batch_size, epochs=num_epochs)

model.save_weights('nn_weights.h5')