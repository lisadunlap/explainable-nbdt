import os
import numpy as np
import torchvision.transforms as transforms
import json
from PIL import Image
from torch.utils.data import Dataset

__all__ = names = ('AnimalsWithAttributes2',)


class AnimalsWithAttributes2(Dataset): 
    train_ratio = 0.8

    def __init__(self, root, transform=None, train=True, download=False, shuffle=False, binary=True, zeroshot=False,
                 ignore_zeroshot_classes=True, **kwargs):
        # load in the data, we ignore test case, only train/val
        self.root = os.path.join(root, 'awa2')
        labels_fname = 'trainclasses.txt' if not zeroshot else 'testclasses.txt'
        matrix_fname = 'predicate-matrix-binary.txt' if binary else 'predicate-matrix-continuous.txt'

        self.photos_path = os.path.join(self.root, 'JPEGImages')
        self.labels_path = os.path.join(self.root, labels_fname)
        self.classes_path = os.path.join(self.root, 'classes.txt')
        self.predicates_path = os.path.join(self.root, 'predicates.txt')
        self.predicates_matrix_path = os.path.join(self.root, matrix_fname)

        self.ignore_zeroshot_classes = ignore_zeroshot_classes
        self.zeroshot = zeroshot
        self.train = train
        self.shuffle = shuffle
        self.transform = transform
        self.load_size = 224
        self.images = []
        self.labels = []
        self.classes = []
        self.use_classes = []
        self.predicates = []
        self.attributes = []

        if download:
            # TODO check if file already exists, otherwise download
            pass

        with open(self.classes_path, 'r') as f:
            for i, line in enumerate(f):
                _, cls = line.strip().split('\t')
                self.classes.append(cls)

        # assuming that file hierarchy is {train/val}/{first letter}/{class}/{fname}.xml
        with open(self.labels_path, 'r') as f: 
            for i, line in enumerate(f):
                label = line.strip()
                self.use_classes.append(label)

        if self.ignore_zeroshot_classes == True:
            self.classes = self.use_classes

        with open(self.predicates_path, 'r') as f:
            for i, line in enumerate(f):
                _, predicate = line.strip().split('\t')
                self.predicates.append(predicate)

        with open(self.predicates_matrix_path, 'r') as f:
            for i, line in enumerate(f):
                predicates = line.strip().split(' ')
                self.attributes.append(predicates)


        np.random.seed(42)
        for image_folder in os.listdir(self.photos_path):
            if image_folder in self.use_classes:
                label = self.classes.index(image_folder)
                image_folder_path = os.path.join(self.photos_path, image_folder)
                new_image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path)]
                np.random.shuffle(new_image_paths)
                n_train = int(len(new_image_paths) * 0.7)
                if self.train:
                    new_image_paths = new_image_paths[:n_train]
                else:
                    new_image_paths = new_image_paths[n_train:]
                self.images.extend(new_image_paths)
                self.labels.extend([label] * len(new_image_paths))            

        if self.shuffle:
            state = np.random.get_state()
            np.random.shuffle(self.labels)
            np.random.set_state(state)
            np.random.shuffle(self.images)

        self.images = np.array(self.images, np.object)
        self.labels = np.array(self.labels, np.int64)
        print("# images found at path '%s': %d" % (self.labels_path, self.images.shape[0]))

        wnid_to_class = self.setup_custom_wnids(root)
        with open(os.path.join(root, 'fake_wnid_to_class.json'), 'w') as f:
            json.dump(wnid_to_class, f)


    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx): 
        image = Image.open(self.images[idx])
        image_np = np.array(image)
        if len(image_np.shape) == 2:
            image_np = np.repeat(image_np[:,:,np.newaxis], 3, axis=2)
            image = Image.fromarray(image_np)
        if self.transform is not None:
            image = self.transform(image)
        # label is the index of the correct category
        label = self.labels[idx]
        predicates = self.attributes[label]
        return (image, predicates), label

    def setup_custom_wnids(self, root):
        wnid_to_class = {}
        with open(os.path.join(self.root, 'wnids.txt'), 'w') as f:
            # use all 9s to avoid conflict with wn
            for i in range(len(self.classes)):
                wnid = 'f%s' % str(i).zfill(8).replace('0', '9')
                wnid_to_class[wnid] = self.classes[i]
                f.write(wnid + '\n')
        return wnid_to_class

    @staticmethod
    def transform_train():
        transform_train = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.Resize((224,224)), # ImageNet standard
            transforms.ToTensor()
        ])
        return transform_train

    @staticmethod
    def transform_test():
        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        return transform_test