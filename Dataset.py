import torch
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import os
import gzip
import numpy as np

class Dataset:
    def __init__(self,data,image_size,batch_size):
        self.data = data
        self.image_size = image_size
        self.batch_size = batch_size


    def load_data(self, path, kind='train'):
        """Load Oracle-MNIST data from `path`"""
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), self.image_size, self.image_size)

        print('The size of %s set: %d'%(kind, len(labels)))

        return images, labels


    def preprocessing(self):
        mean, std = (0.5,), (0.5,)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                        ])

        x_train, y_train = self.load_data(f'{self.data}', kind='train')
        x_test, y_test = self.load_data(f'{self.data}', kind='t10k')

        x_train = x_train.reshape(-1, self.image_size, self.image_size, 1)
        x_test = x_test.reshape(-1, self.image_size, self.image_size, 1)
        x_train_tensor = torch.stack([transform(image.squeeze()) for image in x_train])
        x_test_tensor = torch.stack([transform(image.squeeze()) for image in x_test])

        train_dataset = TensorDataset(x_train_tensor, torch.tensor(y_train))
        test_dataset = TensorDataset(x_test_tensor, torch.tensor(y_test))


        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return trainloader,testloader