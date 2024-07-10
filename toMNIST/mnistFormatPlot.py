import numpy as np
import matplotlib.pyplot as plt

def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        data = f.read()  # Read the rest of the data
        images = np.frombuffer(data, dtype=np.uint8)
        if images.size == num * rows * cols:
            return images.reshape(num, rows, cols)
        else:
            print(f"Expected {num * rows * cols} bytes, got {images.size}")
            return images.reshape(-1, rows, cols)

def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read the header information
        magic, num = np.frombuffer(f.read(8), dtype='>i4')
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Paths to the MNIST-style image and label files
image_file = 'output_256/train-images.idx3-ubyte'
label_file = 'output_256/train-labels.idx1-ubyte'

# Read the images and labels from file
try:
    images = read_mnist_images(image_file)
except ValueError as e:
    print("Error reading MNIST data:", e)
labels = read_mnist_labels(label_file)

# Choose an image index to display
image_index = 10  # Change this to display a different image

# Display the chosen image
plt.imshow(images[image_index], cmap='gray')
plt.title(f'Label: {labels[image_index]}')
plt.show()
