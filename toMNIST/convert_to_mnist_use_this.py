import os
import numpy as np
import imageio
from skimage.transform import resize
from skimage.color import rgb2gray

def convert_images_to_mnist_format(input_folder, output_folder, test_ratio=0.2, img_height=28, img_width=28):
    labels = []
    images = []
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']

    # Iterate over each class directory
    for label, class_folder in enumerate(sorted(os.listdir(input_folder))):
        class_path = os.path.join(input_folder, class_folder)
        if os.path.isdir(class_path):
            for image_filename in os.listdir(class_path):
                if any(image_filename.lower().endswith(ext) for ext in valid_extensions):
                    image_path = os.path.join(class_path, image_filename)
                    try:
                        image = imageio.imread(image_path)
                        image_resized = resize(image, (img_height, img_width), anti_aliasing=True)
                        if image_resized.ndim == 3 and image_resized.shape[2] == 3:
                            image_resized = rgb2gray(image_resized)
                        images.append(image_resized.flatten())
                        labels.append(label)
                    except Exception as e:
                        print(f"Failed to process {image_path}: {e}")

    images_np = np.array(images, dtype=np.uint8)
    labels_np = np.array(labels, dtype=np.uint8)

    # Shuffle the dataset
    indices = np.arange(len(labels_np))
    np.random.shuffle(indices)
    images_np = images_np[indices]
    labels_np = labels_np[indices]

    # Splitting dataset into training and testing
    split_index = int(len(images_np) * (1 - test_ratio))
    train_images, test_images = images_np[:split_index], images_np[split_index:]
    train_labels, test_labels = labels_np[:split_index], labels_np[split_index:]

    # Save training and testing data in MNIST format
    save_mnist_format(train_images, train_labels, os.path.join(output_folder, 'train-images.idx3-ubyte'), os.path.join(output_folder, 'train-labels.idx1-ubyte'))
    save_mnist_format(test_images, test_labels, os.path.join(output_folder, 't10k-images.idx3-ubyte'), os.path.join(output_folder, 't10k-labels.idx1-ubyte'))

def save_mnist_format(images, labels, image_path, label_path):
    # Save image data
    print(len(images))
    image_header = np.array([0x0803, len(images), 224, 224], dtype='>i4')
    with open(image_path, "wb") as f:
        f.write(image_header.tobytes())
        f.write(images.tobytes())
    
    # Save label data
    label_header = np.array([0x0801, len(labels)], dtype='>i4')
    with open(label_path, "wb") as f:
        f.write(label_header.tobytes())
        f.write(labels.tobytes())

# Example usage
input_folder = '/Users/yifangbai/Desktop/HUST-OBS/deciphered'
output_folder = 'output_224'
test_ratio = 0.2  # 20% of the data will be used for testing
convert_images_to_mnist_format(input_folder, output_folder, test_ratio, 224, 224)
