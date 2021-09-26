import numpy as np
INPUT_SIZE = 1200
OUTPUT_SIZE = 5

def process_text_entry(filename):
    inputs = []
    batch_size = 0
    with open(filename, 'r') as file:
        count = 0
        for question in file:
            question = question.rstrip('\n')
            # Changing characters into their ASCII #
            input = [ord(x) for x in question]
            if len(input) < INPUT_SIZE:
                input = input + [0 for i in range(INPUT_SIZE - len(input))]
            input = np.array(input)
            inputs.append(input)
            count += 1
        batch_size = count
        inputs = np.array(inputs)

    return inputs, batch_size

def format_labels(filename):
    expected_outputs = []
    with open(filename, 'r') as file:
        for label in file:
            label = label.rstrip('\n')
            output = make_output_format(label)
            expected_outputs.append(output)
        expected_outputs = np.array(expected_outputs)
    return expected_outputs
            
            
def make_output_format(label):
    template = ['Fis', 'Mat', 'Qui', 'Port', 'Ing']
    output = np.array([0, 0, 0, 0, 0])
    if label in template:
        output[template.index(label)] = 1
    else:
        print('Label {} not defined'.format(label))
    return output

if __name__ == '__main__':
    process_text_entry('Thimot/Teste (copy).txt')