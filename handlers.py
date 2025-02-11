import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class MNIST_Handler(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x.numpy(), mode='L')
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class SVHN_Handler(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(np.transpose(x, (1, 2, 0)))
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class CIFAR10_Handler(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        #x = np.clip(x, 0, 1)
        #x = (x * 255).astype(np.uint8)
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

import torch

class UC_Merced_Handler(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform

    # def __getitem__(self, index):
    #     x, y = self.X[index], self.Y[index]
    #     x = Image.fromarray(x.numpy(), mode='RGB')
    #     # if x.ndim == 3:  # Check if x is a 3D array
    #     #     x =  x.to(torch.float32).mean(axis=2)  # Convert to grayscale by averaging the channels
    #     # x = Image.fromarray(x.numpy(), mode='L')
    #     # print(x.size)
    #     x = self.transform(x)
    #     return x, y, index
    
    def __getitem__(self, idx):
        img_path = self.X[idx]
        label = self.Y[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label, idx
    
    def __len__(self):
        return len(self.X)
    
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class UC_Merced_Imb_Handler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x, mode='RGB')  # Remove .numpy() since x is already a NumPy array
        if self.transform:
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class openml_Handler(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)

class MNIST_Handler_joint(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:

            x_1 = Image.fromarray(x_1.numpy(), mode='L')
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(x_2.numpy(), mode='L')
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2

class SVHN_Handler_joint(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform = None):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:

            x_1 = Image.fromarray(np.transpose(x_1, (1, 2, 0)))
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(np.transpose(x_2, (1, 2, 0)))
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2

class CIFAR10_Handler_joint(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform = None):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:

            x_1 = Image.fromarray(x_1)
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(x_2)
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2