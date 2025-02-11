import numpy as np
import torch
import random
import os
from torchvision import datasets
from PIL import Image
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler, args_task):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        self.args_task = args_task
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_unlabeled_data_by_idx(self, idx):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs][idx]
    
    def get_data_by_idx(self, idx):
        return self.X_train[idx], self.Y_train[idx]

    def get_new_data(self, X, Y):
        return self.handler(X, Y, self.args_task['transform_train'])

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs], self.args_task['transform_train'])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs], self.args_task['transform_train'])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, self.args_task['transform_train'])

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test, self.args_task['transform'])
    
    def get_partial_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return self.X_train[labeled_idxs], self.Y_train[labeled_idxs]
    
    def get_partial_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs]

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    
def get_MNIST(handler, args_task):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)

def get_FashionMNIST(handler, args_task):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)

def get_EMNIST(handler, args_task):
    raw_train = datasets.EMNIST('./data/EMNIST', split = 'byclass', train=True, download=True)
    raw_test = datasets.EMNIST('./data/EMNIST', split = 'byclass', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)

def get_SVHN(handler, args_task):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data, torch.from_numpy(data_train.labels), data_test.data, torch.from_numpy(data_test.labels), handler, args_task)

def get_CIFAR10(handler, args_task):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data, torch.LongTensor(data_train.targets), data_test.data, torch.LongTensor(data_test.targets), handler, args_task)

def get_CIFAR10_imb(handler, args_task):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    X_tr = data_train.data
    Y_tr = torch.from_numpy(np.array(data_train.targets)).long()
    X_te = data_test.data
    Y_te = torch.from_numpy(np.array(data_test.targets)).long()
    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    X_tr_imb = []
    Y_tr_imb = []
    random.seed(4666)
    for i in range(Y_tr.shape[0]):
        tmp = random.random()
        if tmp < ratio[Y_tr[i]]:
            X_tr_imb.append(X_tr[i])
            Y_tr_imb.append(Y_tr[i])
    X_tr_imb = np.array(X_tr_imb).astype(X_tr.dtype)
    Y_tr_imb = torch.LongTensor(np.array(Y_tr_imb)).type_as(Y_tr)
    return Data(X_tr_imb, Y_tr_imb, X_te, Y_te, handler, args_task)

def get_CIFAR100(handler, args_task):
    data_train = datasets.CIFAR100('./data/CIFAR100', train=True, download=True)
    data_test = datasets.CIFAR100('./data/CIFAR100', train=False, download=True)
    return Data(data_train.data, torch.LongTensor(data_train.targets), data_test.data, torch.LongTensor(data_test.targets), handler, args_task)

def get_TinyImageNet(handler, args_task):
    import cv2
    #download data from http://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it into ./data/TinyImageNet
    # deal with training set
    Y_train_t = []
    train_img_names = []
    train_imgs = []
    
    with open('./data/TinyImageNet/tiny-imagenet-200/wnids.txt') as wnid:
        for line in wnid:
            Y_train_t.append(line.strip('\n'))
    for Y in Y_train_t:
        Y_path = './data/TinyImageNet/tiny-imagenet-200/train/' + Y + '/' + Y + '_boxes.txt'
        train_img_name = []
        with open(Y_path) as Y_p:
            for line in Y_p:
                train_img_name.append(line.strip('\n').split('\t')[0])
        train_img_names.append(train_img_name)
    train_labels = np.arange(200)
    idx = 0
    for Y in Y_train_t:
        train_img = []
        for img_name in train_img_names[idx]:
            img_path = os.path.join('./data/TinyImageNet/tiny-imagenet-200/train/', Y, 'images', img_name)
            train_img.append(cv2.imread(img_path))
        train_imgs.append(train_img)
        idx = idx + 1
    train_imgs = np.array(train_imgs)
    train_imgs = train_imgs.reshape(-1, 64, 64, 3)
    X_tr = []
    Y_tr = []
    for i in range(train_imgs.shape[0]):
        Y_tr.append(i//500)
        X_tr.append(train_imgs[i])
    #X_tr = torch.from_numpy(np.array(X_tr))
    X_tr = np.array(X_tr)
    Y_tr = torch.from_numpy(np.array(Y_tr)).long()

    #deal with testing (val) set
    Y_test_t = []
    Y_test = []
    test_img_names = []
    test_imgs = []
    with open('./data/TinyImageNet/tiny-imagenet-200/val/val_annotations.txt') as val:
        for line in val:
            test_img_names.append(line.strip('\n').split('\t')[0])
            Y_test_t.append(line.strip('\n').split('\t')[1])
    for i in range(len(Y_test_t)):
        for i_t in range(len(Y_train_t)):
            if Y_test_t[i] == Y_train_t[i_t]:
                Y_test.append(i_t)
    test_labels = np.array(Y_test)
    test_imgs = []
    for img_name in test_img_names:
        img_path = os.path.join('./data/TinyImageNet/tiny-imagenet-200/val/images', img_name)
        test_imgs.append(cv2.imread(img_path))
    test_imgs = np.array(test_imgs)
    X_te = []
    Y_te = []

    for i in range(test_imgs.shape[0]):
        X_te.append(test_imgs[i])
        Y_te.append(Y_test[i])
    #X_te = torch.from_numpy(np.array(X_te))
    X_te = np.array(X_te)
    Y_te = torch.from_numpy(np.array(Y_te)).long()
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

def get_openml(handler, args_task, selection = 6):
    import openml
    from sklearn.preprocessing import LabelEncoder
    openml.config.apikey = '3411e20aff621cc890bf403f104ac4bc'
    openml.config.set_cache_directory('./data/openml/')
    ds = openml.datasets.get_dataset(selection)
    data = ds.get_data(target=ds.default_target_attribute)
    X = np.asarray(data[0])
    y = np.asarray(data[1])
    y = LabelEncoder().fit(y).transform(y)

    num_classes = int(max(y) + 1)
    nSamps, _ = np.shape(X)
    testSplit = .1
    inds = np.random.permutation(nSamps)
    X = X[inds]
    y = y[inds]

    split =int((1. - testSplit) * nSamps)
    while True:
        inds = np.random.permutation(split)
        if len(inds) > 50000: inds = inds[:50000]
        X_tr = X[:split]
        X_tr = X_tr[inds]
        X_tr = torch.Tensor(X_tr)

        y_tr = y[:split]
        y_tr = y_tr[inds]
        Y_tr = torch.Tensor(y_tr).long()

        X_te = torch.Tensor(X[split:])
        Y_te = torch.Tensor(y[split:]).long()

        if len(np.unique(Y_tr)) == num_classes: break
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

def get_BreakHis(handler, args_task):
    # download data from https://www.kaggle.com/datasets/ambarish/breakhis and unzip it in data/BreakHis/
    data_dir = './data/BreakHis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'
    data = datasets.ImageFolder(root = data_dir, transform = None).imgs
    train_ratio = 0.7
    test_ratio = 0.3
    data_idx = list(range(len(data)))
    random.shuffle(data_idx)
    train_idx = data_idx[:int(len(data)*train_ratio)]
    test_idx = data_idx[int(len(data)*train_ratio):]
    X_tr = [np.array(Image.open(data[i][0])) for i in train_idx]
    Y_tr = [data[i][1] for i in train_idx]
    X_te = [np.array(Image.open(data[i][0])) for i in test_idx]
    Y_te = [data[i][1] for i in test_idx]
    X_tr = np.array(X_tr, dtype=object)
    X_te = np.array(X_te, dtype=object)
    Y_tr = torch.from_numpy(np.array(Y_tr))
    Y_te = torch.from_numpy(np.array(Y_te))
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

def get_PneumoniaMNIST(handler, args_task):
    # download data from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia and unzip it in data/PhwumniaMNIST/
    import cv2

    data_train_dir = './data/PneumoniaMNIST/chest_xray/train/'
    data_test_dir = './data/PneumoniaMNIST/chest_xray/test/'
    assert os.path.exists(data_train_dir)
    assert os.path.exists(data_test_dir)

    #train data
    train_imgs_path_0 = [data_train_dir+'NORMAL/'+f for f in os.listdir(data_train_dir+'/NORMAL/')]
    train_imgs_path_1 = [data_train_dir+'PNEUMONIA/'+f for f in os.listdir(data_train_dir+'/PNEUMONIA/')]
    train_imgs_0 = []
    train_imgs_1 = []
    for p in train_imgs_path_0:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        train_imgs_0.append(im)
    for p in train_imgs_path_1:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        train_imgs_1.append(im)
    train_labels_0 = np.zeros(len(train_imgs_0))
    train_labels_1 = np.ones(len(train_imgs_1))
    X_tr = []
    Y_tr = []
    train_imgs = train_imgs_0 + train_imgs_1
    train_labels = np.concatenate((train_labels_0, train_labels_1))
    idx_train = list(range(len(train_imgs)))
    random.seed(4666)
    random.shuffle(idx_train)
    X_tr = [train_imgs[i] for i in idx_train]
    Y_tr = [train_labels[i] for i in idx_train]
    X_tr = np.array(X_tr)
    Y_tr = torch.from_numpy(np.array(Y_tr)).long()

    #test data
    test_imgs_path_0 = [data_test_dir+'NORMAL/'+f for f in os.listdir(data_test_dir+'/NORMAL/')]
    test_imgs_path_1 = [data_test_dir+'PNEUMONIA/'+f for f in os.listdir(data_test_dir+'/PNEUMONIA/')]
    test_imgs_0 = []
    test_imgs_1 = []
    for p in test_imgs_path_0:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        test_imgs_0.append(im)
    for p in test_imgs_path_1:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        test_imgs_1.append(im)
    test_labels_0 = np.zeros(len(test_imgs_0))
    test_labels_1 = np.ones(len(test_imgs_1))
    X_te = []
    Y_te = []
    test_imgs = test_imgs_0 + test_imgs_1
    test_labels = np.concatenate((test_labels_0, test_labels_1))
    idx_test = list(range(len(test_imgs)))
    random.seed(4666)
    random.shuffle(idx_test)
    X_te = [test_imgs[i] for i in idx_test]
    Y_te = [test_labels[i] for i in idx_test]
    X_te = np.array(X_te)
    Y_te = torch.from_numpy(np.array(Y_te)).long()

    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

def get_waterbirds(handler, args_task):
    import wilds
    from torchvision import transforms
    dataset = wilds.get_dataset(dataset='waterbirds', root_dir='./data/waterbirds', download='True')
    trans = transforms.Compose([transforms.Resize([255,255])])
    train = dataset.get_subset(split = 'train',transform = trans)
    test = dataset.get_subset(split = 'test', transform = trans)

    len_train = train.metadata_array.shape[0]
    len_test = test.metadata_array.shape[0]
    X_tr = []
    Y_tr = []
    X_te = []
    Y_te = []

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    f = open('waterbirds.txt', 'w')

    for i in range(len_train):
        x,y,meta = train.__getitem__(i)
        img = np.array(x)
        X_tr.append(img)
        Y_tr.append(y)

    for i in range(len_test):
        x,y, meta = test.__getitem__(i)
        img = np.array(x)

        X_te.append(img)
        Y_te.append(y)
        if meta[0] == 0 and meta[1] == 0:
            f.writelines('1') #landbird_background:land
            f.writelines('\n')
            count1 = count1 + 1
        elif meta[0] == 1 and meta[1] == 0:
            f.writelines('2') #landbird_background:water
            count2 = count2 + 1
            f.writelines('\n')
        elif meta[0] == 0 and meta[1] == 1:
            f.writelines('3') #waterbird_background:land
            f.writelines('\n')
            count3 = count3 + 1
        elif meta[0] == 1 and meta[1] == 1:
            f.writelines('4') #waterbird_background:water
            f.writelines('\n')
            count4 = count4 + 1
        else:
            raise NotImplementedError    
    f.close()

    Y_tr = torch.tensor(Y_tr)
    Y_te = torch.tensor(Y_te)
    X_tr = np.array(X_tr)
    X_te = np.array(X_te)
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

from torchgeo.datasets import UCMerced
from torchvision.datasets import ImageFolder
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import make_dataset

# ---------------------- UC Merced -----------------------
def print_class_distribution(Y, name):
    unique, counts = np.unique(Y, return_counts=True)
    print(f"Class distribution in {name}:")
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} samples")

def resize_image(image_array, target_size):
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image_array)
    # Resize the image
    resized_image = image.resize(target_size, Image.ANTIALIAS)
    # Convert the PIL Image back to a NumPy array
    return np.array(resized_image)

def find_classes(directory) :
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

# def get_UcMerced(handler, args_task):
#     train_dataset = UCMerced('./data/UCMerced_LandUse/Images/', split='train', download=True)
#     test_dataset = UCMerced('./data/UCMerced_LandUse/Images/', split='test',  download=True)
    
#     target_size = (256, 256)
#     dataset = ImageFolder(root='./data/UCMerced_LandUse/Images/UCMerced_LandUse/Images/')
#     classes = dataset.classes
    
#     directory = os.path.expanduser('./data/UCMerced_LandUse/Images/UCMerced_LandUse/Images/')

#     _, class_to_idx = find_classes(directory)
    
#     # Use make_dataset to create the samples list
#     samples = make_dataset('./data/UCMerced_LandUse/Images/UCMerced_LandUse/Images/', class_to_idx, extensions=('.tif', '.tiff'))
    
#     # Print the samples
#     # for img_path, label in samples:
#     #     print(f"Image Path: {img_path}, Label: {label}")

#     #######################################################
#     ## Clean the wrong_shape files
#     #######################################################
#     with open('wrong_shape.txt', 'w') as f:
#         f.write('')
#     #
#     #######################################################

#     # Initialize lists to store image arrays
#     train_images = []
#     valid_images = []
#     test_images = []

#     train_targets = []
#     valid_targets = []
#     test_targets = []

#     import tifffile as tiff

#     folder_path = './data/UCMerced_LandUse/Images/UCMerced_LandUse/Images/'

#     # TEST file 
#     with open('./data/UCMerced_LandUse/Images/uc_merced-test.txt', 'r') as file:
#         # Iterate through each line in the file
#         for line in file:
#             img_name = line.strip()
#             class_name = img_name[:-6]
#             img_array_test = tiff.imread( folder_path + class_name + '/' + img_name)
#             #print(img_array_test)
#             if img_array_test.shape != (256, 256,3): 
#                 # write in a .txt file the name of the image that has a different shape
#                 with open('wrong_shape.txt', 'a') as f:
#                     str_ = 'TEST,'+ img_name + ','+ str(img_array_test.shape)+'\n'
#                     f.write(str_)
#             resized_img_array_test = resize_image(img_array_test, target_size)
#             test_images.append(resized_img_array_test)
#             test_targets.append(class_to_idx[class_name])

#     # TRAIN file
#     with open('./data/UCMerced_LandUse/Images/uc_merced-train.txt', 'r') as file:
#         # Iterate through each line in the file
#         for line in file:
#             img_name = line.strip()
#             class_name = img_name[:-6]
#             img_array_train = tiff.imread( folder_path + class_name + '/' + img_name)
#             # print(img_array_train)
#             #print(img_array_train.shape)
#             if img_array_train.shape != (256, 256,3): 
#                 # write in a .txt file the name of the image that has a different shape
#                 with open('wrong_shape.txt', 'a') as f:
#                     str_ = 'TRAIN,'+ img_name + ','+ str(img_array_train.shape)+'\n'
#                     f.write(str_)
#             resized_img_array_train = resize_image(img_array_train, target_size)
#             train_images.append(resized_img_array_train)
#             train_targets.append(class_to_idx[class_name])
#     #print_class_distribution(train_targets, "training set")

#     # VALID file
#     with open('./data/UCMerced_LandUse/Images/uc_merced-val.txt', 'r') as file:
#         # Iterate through each line in the file
#         for line in file:
#             img_name = line.strip()
#             class_name = img_name[:-6]
#             img_array_valid = tiff.imread( folder_path + class_name + '/' + img_name)
#             # print(img_array_valid)
#             if img_array_valid.shape != (256, 256,3): 
#                 # write in a .txt file the name of the image that has a different shape
#                 with open('wrong_shape.txt', 'a') as f:
#                     str_ = 'VAL,'+ img_name + ','+ str(img_array_valid.shape)+'\n'
#                     f.write(str_)
#             resized_img_array_valid = resize_image(img_array_valid, target_size)
#             valid_images.append(resized_img_array_valid)
#             valid_targets.append(class_to_idx[class_name])

#     print(class_to_idx)
#     test_images_tensor = torch.tensor(test_images)
#     train_images_tensor = torch.tensor(train_images)
#     valid_images_tensor = torch.tensor(valid_images)

#     # print number of classes from the train_targets
#     #num_classes = len(set(train_targets))
#     #print(f"Number of classes in the training set: {num_classes}")
    
#     return Data(train_images_tensor, torch.tensor(train_targets), test_images_tensor, torch.tensor(test_targets), handler, args_task)


import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# def get_UC_Merced(handler, args_task):
def get_UcMerced_2(handler, args_task):
    data_dir = './data/UCMerced_LandUse/Images/UCMerced_LandUse/Images'
    class_names = sorted(os.listdir(data_dir))
    
    # Collect all file paths and labels
    file_paths = []
    labels = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.tif'):
                    file_paths.append(os.path.join(class_dir, file_name))
                    labels.append(label_idx)

    # Split data into train/test
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    total_samples = len(file_paths)
    indices = np.random.permutation(total_samples)
    split = int(total_samples * 0.8)  # 80% train, 20% test
    
    train_indices = indices[:split]
    test_indices = indices[split:]

    X_train = file_paths[train_indices]
    Y_train = torch.tensor(labels[train_indices], dtype=torch.long)  # Convert to tensor
    X_test = file_paths[test_indices]
    Y_test = torch.tensor(labels[test_indices], dtype=torch.long)  # Convert to tensor

    # Return a Data instance
    return Data(X_train, Y_train, X_test, Y_test, handler, args_task)

""" Function that loads the UCMerced dataset and creates an imbalanced dataset with the specified imbalance ratios. """
def get_UcMerced_Imbalanced_2(handler, args_task, imbalance_ratios=None):
    data_dir = './data/UCMerced_LandUse/Images/UCMerced_LandUse/Images'
    class_names = sorted(os.listdir(data_dir))

    # Collect all file paths and labels
    file_paths = []
    labels = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.tif'):
                    file_paths.append(os.path.join(class_dir, file_name))
                    labels.append(label_idx)
    # Split data into train/test
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    total_samples = len(file_paths)
    indices = np.random.permutation(total_samples)
    split = int(total_samples * 0.8)  # 80% train, 20% test

    train_indices = indices[:split]
    test_indices = indices[split:]

    X_train = file_paths[train_indices]
    Y_train = labels[train_indices]
    X_test = file_paths[test_indices]
    Y_test = labels[test_indices]

    # Create imbalance if imbalance_ratios is provided
    if imbalance_ratios:
        class_data = {label: [] for label in np.unique(Y_train)}
        for x, y in zip(X_train, Y_train):
            class_data[y].append(x)

        imbalanced_class_data = create_imbalance(class_data, imbalance_ratios)

        X_train = []
        Y_train = []
        for label, samples in imbalanced_class_data.items():
            X_train.extend(samples)
            Y_train.extend([label] * len(samples))

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
    
    # Apply oversampling to balance the dataset
    X_train_resampled, Y_train_resampled = oversample_data(X_train, Y_train)

    # Convert to tensors
    Y_train = torch.tensor(Y_train_resampled, dtype=torch.long)
    Y_test = torch.tensor(Y_test, dtype=torch.long)
    #print class distribution
    print_class_distribution(Y_train_resampled, "training set -oversampled")
    
    # Return a Data instance
    return Data(X_train_resampled, Y_train_resampled, X_test, Y_test, handler, args_task)

import os
from PIL import Image
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler
import random

def create_imbalance(data, imbalance_ratios):
    """
    Create an imbalanced dataset by reducing the number of samples for certain classes.
    
    Parameters:
    - data: A dictionary with class labels as keys and lists of samples as values.
    - imbalance_ratios: A dictionary with class labels as keys and imbalance ratios as values.
    
    Returns:
    - imbalanced_data: A dictionary with the imbalanced dataset.
    """
    imbalanced_data = {}
    for class_label, samples in data.items():
        if class_label in imbalance_ratios:
            ratio = imbalance_ratios[class_label]
            reduced_samples = random.sample(samples, int(len(samples) * ratio))
            imbalanced_data[class_label] = reduced_samples
        else:
            imbalanced_data[class_label] = samples
    print("Class distribution after imbalance:")
    for class_label, samples in imbalanced_data.items():
        print(f"Class {class_label}: {len(samples)} samples")
    return imbalanced_data

def oversample_data(X, Y):
    """
    Oversample the minority classes in the dataset to balance the class distribution.
    
    Parameters:
    - X: List of data samples.
    - Y: List of labels corresponding to the data samples.
    
    Returns:
    - X_resampled: List of resampled data samples.
    - Y_resampled: List of resampled labels.
    """

    # Convert lists to numpy arrays for compatibility with imblearn
    X = np.array(X)
    Y = np.array(Y)
    # Reshape X to 2D array for oversampling
    X_reshaped = X.reshape((X.shape[0], -1))
    # Apply RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_resampled, Y_resampled = ros.fit_resample(X_reshaped, Y)
    # Reshape X_resampled back to original shape
    X_resampled = X_resampled.reshape((X_resampled.shape[0],) + X.shape[1:])

    return X_resampled, Y_resampled


def undersample_data(X, Y):
    """
    Undersample the majority classes in the dataset to balance the class distribution.
    
    Parameters:
    - X: List of data samples.
    - Y: List of labels corresponding to the data samples.
    
    Returns:
    - X_resampled: List of resampled data samples.
    - Y_resampled: List of resampled labels.
    """
    # Convert lists to numpy arrays for compatibility with imblearn
    X = np.array(X)
    Y = np.array(Y)
    # Reshape X to 2D array for oversampling
    X_reshaped = X.reshape((X.shape[0], -1))
    # Apply RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_resampled, Y_resampled = ros.fit_resample(X_reshaped, Y)
    # Reshape X_resampled back to original shape
    X_resampled = X_resampled.reshape((X_resampled.shape[0],) + X.shape[1:])

    return X_resampled, Y_resampled

def get_UCMerced_w_imbalance(handler, args_task, imbalance_ratios, transform=None):
    """
    Load the UC Merced dataset, create an imbalanced dataset, and split it into train, validation, and test sets.
    
    Parameters:
    - handler: A function to handle the data transformation.
    - args_task: Arguments related to the task, including transformations.
    - imbalance_ratios: A dictionary with class labels as keys and imbalance ratios as values.
    
    Returns:
    - train_data: Training data.
    - val_data: Validation data.
    - test_data: Test data.
    """
    # Set default transform if not provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    # Load the dataset (this is a placeholder, replace with actual loading code)
    dataset = load_uc_merced_dataset("./data/UCMerced_LandUse/Images/UCMerced_LandUse/Images/")
    
    #data = get_UcMerced(handler, args_task)
    # ---by using get_UcMerced(handler, args_task)---
    # X_train: Tensor,
    # Y_train: Tensor,
    # X_test: Tensor,
    # Y_test: Tensor,
    # handler: Any,
    # args_task: Any
    # -----------------------------------------------
    # Split the dataset into classes
    class_data = {}
    for sample in dataset:
        class_label = sample['label']
        if class_label not in class_data:
            class_data[class_label] = []
        class_data[class_label].append(sample)
    
    # Create an imbalanced dataset
    imbalanced_data = create_imbalance(class_data, imbalance_ratios)
    # Flatten the imbalanced data
    imbalanced_samples = []
    for class_label, samples in imbalanced_data.items():
        imbalanced_samples.extend(samples)
    
    # Extract features and labels
    X = [sample['data'] for sample in imbalanced_samples]
    Y = [sample['label'] for sample in imbalanced_samples]

     # Apply transformations to ensure consistent image shapes
    X = [transform(Image.fromarray(x, mode='RGB')).numpy() for x in X]
    
    # Split the imbalanced data into train, validation, and test sets
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42, stratify=Y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Convert to numpy arrays and tensors
    X_train = np.array(X_train)
    #X_val = np.array(X_val)
    X_test = np.array(X_test)
    Y_train = torch.tensor(Y_train)
    #Y_val = torch.tensor(Y_val)
    Y_test = torch.tensor(Y_test)

     # Create dataset handlers
    train_handler = handler(X_train, Y_train, transform)
    val_handler = handler(X_val, Y_val, transform)
    test_handler = handler(X_test, Y_test, transform)

    return Data(X_train, Y_train, X_test, Y_test, handler, args_task)

    

def load_uc_merced_dataset(data_path):
        """
        Load the UC Merced dataset from the specified path.
        
        Parameters:
        - data_path: Path to the dataset.
        
        Returns:
        - dataset: A list of dictionaries with 'data' and 'label' keys.
        """
        dataset = []
        class_names = sorted(os.listdir(data_path))
        target_size = (256, 256)
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_path, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        img_array = np.array(img)
                        if img_array.shape[:2] != target_size:
                            img_array = resize_image(img_array, target_size)
                        dataset.append({'data': img_array, 'label': class_idx})
        print(f"Loaded dataset with {len(dataset)} samples.")
        return dataset
    # dataset = []
    # class_names = sorted(os.listdir(data_path))
    
    # for class_idx, class_name in enumerate(class_names):
    #     class_dir = os.path.join(data_path, class_name)
    #     if os.path.isdir(class_dir):
    #         for img_name in os.listdir(class_dir):
    #             img_path = os.path.join(class_dir, img_name)
    #             with Image.open(img_path) as img:
    #                 img = img.convert('RGB')
    #                 img_array = np.array(img)
    #                 dataset.append({'data': img_array, 'label': class_idx})
    
    # return dataset





#############################################################################
# class CustomUCMerced(UCMerced):
#     def __init__(self, root, split, transform=None, download=False):
#         super().__init__(root, split=split, download=download)
#         self.transform = transform

#     def __getitem__(self, index):
#         sample = super().__getitem__(index)
#         image = sample['image']
#         label = sample['label']
        
#         if self.transform:
#             image = self.transform(image)
        
#         return {'image': image, 'label': label}

# def get_UcMerced(handler, args_task):
#     #trans = transforms.Compose([transforms.Resize([256, 256])])
#     train_dataset = CustomUCMerced('./data/UCMerced_LandUse/Images/', split='train', download=True)
#     test_dataset = CustomUCMerced('./data/UCMerced_LandUse/Images/', split='test',  download=True)
    
#     print('train_dataset:', (train_dataset))
#     print('test_dataset:', (test_dataset))