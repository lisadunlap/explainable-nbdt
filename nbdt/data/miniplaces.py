import os
import torchsample
import torchvision.transforms as transforms
from torch.utils.data import Dataset

__all__ = names = ('MiniPlaces',)



class MiniPlaces(Dataset): 
    def __init__(self, root, transform, train=True, download=False, **kwargs):
        # load in the data, we ignore test case, only train/val
        self.root = os.path.join(root, 'miniplaces')
        labels_fname = 'train.txt' if train else 'val.txt'
        photos_path = os.path.join(self.root, 'images')
        labels_path = os.path.join(self.root, labels_fname)
        categories_path = os.path.join(self.root, 'categories.txt')

        self.photos_path = photos_path
        self.labels_path = labels_path
        self.categories_path = categories_path
        self.transform = transform
        self.load_size = 224
        self.images = []
        self.labels = []
        self.classes = []

        if download:
            # TODO check if file already exists, otherwise download
            pass

        # read the file
        # assuming that file hierarchy is {train/val}/{first letter}/{class}/{fname}.xml
        with open(self.labels_path, 'r') as f: 
            for line in f:
                path, label = line.strip().split(" ")
                self.images.append(path)
                self.labels.append(label)

        with open(self.categories_path, 'r') as f:
            for line in f:
                cls, _ = line.strip().split(" ")
                self.classes.append(cls)

        self.images = np.array(self.images, np.object)
        self.labels = np.array(self.labels, np.int64)
        print("# images found at path '%s': %d" % (self.labels_path, self.images.shape[0]))

        wnid_to_class = self.setup_custom_wnids(root)
        with open(os.path.join(root, 'fake_wnid_to_class.json'), 'w') as f:
            json.dump(wnid_to_class, f)

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx): 
        image = Image.open(os.path.join(self.photos_path, self.images[idx]))
        image = self.transform(image)
        # label is the index of the correct category
        label = self.labels[idx]
        return (image, label)

    def setup_custom_wnids(self, root):
        wnid_to_class = {}
        with open(os.path.join(self.root, 'wnids.txt'), 'w') as f:
            # use all 9s to avoid conflict with wn
            for i in range(100):
                wnid = 'f%s' % str(i).zfill(8).replace('0', '9')
                wnid_to_class[wnid] = self.classes[i]
                f.write(wnid + '\n')
        return wnid_to_class

    @staticmethod
    def transform_train():
        transform_train = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchsample.transforms.RandomRotate(30),
            torchsample.transforms.RandomGamma(0.5, 1.5),
            torchsample.transforms.RandomSaturation(-0.8, 0.8),
            torchsample.transforms.RandomBrightness(-0.3, 0.3),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform_train

    @staticmethod
    def transform_test():
        transform_test = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform_test

