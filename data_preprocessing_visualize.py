import os
import torch
import matplotlib.pyplot as plt
import numpy as np


'''Visualization of dataset of flower activity  affected/not affected by motions around
work data_preprocessing.py results: labeled_data.pth file'''

OSCILLO_FOLDER = 'dataset/2023-11-29'

if __name__ == '__main__':
    current_directory = os.getcwd()
    load_path = os.path.join(current_directory, OSCILLO_FOLDER, 'labeled_data.pth')
    load_data = torch.load(load_path)
    x_tensor = load_data['labeled_voltage_v1'].unsqueeze(2)
    label_tensor = load_data['labels']
    unique_classes, counts = torch.unique(label_tensor, return_counts=True)
    print('samples by class:',counts)
    class_0_indices = (label_tensor == 0).nonzero(as_tuple=True)[0]
    class_1_indices = (label_tensor == 1).nonzero(as_tuple=True)[0]


    # Choose a random index for each class
    random_index_class_0 = np.random.choice(class_0_indices)
    random_index_class_1 = np.random.choice(class_1_indices)

    # Extract the corresponding sequences
    sequence_class_0 = x_tensor[random_index_class_0].squeeze().numpy()
    sequence_class_1 = x_tensor[random_index_class_1].squeeze().numpy()

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Plot for class 0
    axs[0].plot(sequence_class_0)
    axs[0].set_title('Random Sample from Class 0')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')

    # Plot for class 1
    axs[1].plot(sequence_class_1)
    axs[1].set_title('Random Sample from Class 1')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Amplitude')

    # Display the plots
    plt.tight_layout()
    plt.show()