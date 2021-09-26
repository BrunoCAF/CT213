import os
import numpy as np
from neural_network import make_neural_network
from utils import process_text_entry, format_labels

def translate_label(output):
    pos = np.argmax(output)
    labels = ['Fis', 'Mat', 'Qui', 'Port', 'Ing']
    return labels[pos]


OUTPUT_SIZE = 5

model = make_neural_network(OUTPUT_SIZE)

if os.path.exists('nn_weights.h5'):
    print('Loading weights from previous learning session.')
    model.load_weights("nn_weights.h5")
else:
    print('No weights found from previous learning session. Unable to proceed.')
    exit(-1)

inputs_predict, batch_size = process_text_entry('Evaluate/Questions.txt')
expected_outputs = format_labels('Evaluate/Tags.txt')

outputs = model.predict(inputs_predict)

for i in range(batch_size):
    label = translate_label(outputs[i])
    expected_label = translate_label(expected_outputs[i])
    if expected_label == label:
        answer = 'correct'
    else:
        answer = 'wrong'
    print('Questão {} => {} {} ({})'.format(i+1, label, answer, expected_label))