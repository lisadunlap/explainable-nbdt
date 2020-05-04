import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import zipfile
import urllib.request
import shutil



__all__ = names = ('TinyImagenet200', 'Imagenet1000','MiniImagenet')


class TinyImagenet200(Dataset):
    """Tiny imagenet dataloader"""

    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    dataset = None

    def __init__(self, root='./data',
            *args, train=True, download=False, **kwargs):
        super().__init__()

        if download:
            self.download(root=root)
        dataset = _TinyImagenet200Train if train else _TinyImagenet200Val
        self.root = root
        self.dataset = dataset(root, *args, **kwargs)
        self.classes = self.dataset.classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    @staticmethod
    def transform_train(input_size=64):
        return transforms.Compose([
            transforms.RandomCrop(input_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])

    @staticmethod
    def transform_val(input_size=-1):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])

    def download(self, root='./'):
        """Download and unzip Imagenet200 files in the `root` directory."""
        dir = os.path.join(root, 'tiny-imagenet-200')
        dir_train = os.path.join(dir, 'train')
        if os.path.exists(dir) and os.path.exists(dir_train):
            print('==> Already downloaded.')
            return

        print('==> Downloading TinyImagenet200...')
        file_name = os.path.join(root, 'tiny-imagenet-200.zip')

        with urllib.request.urlopen(self.url) as response, open(file_name, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            print('==> Extracting TinyImagenet200...')
            with zipfile.ZipFile(file_name) as zf:
                zf.extractall(root)

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class _TinyImagenet200Train(datasets.ImageFolder):

    def __init__(self, root='./data', *args, **kwargs):
        super().__init__(os.path.join(root, 'tiny-imagenet-200/train'), *args, **kwargs)


class _TinyImagenet200Val(datasets.ImageFolder):

    def __init__(self, root='./data', *args, **kwargs):
        super().__init__(os.path.join(root, 'tiny-imagenet-200/val'), *args, **kwargs)

        self.path_to_class = {}
        with open(os.path.join(self.root, 'val_annotations.txt')) as f:
            for line in f.readlines():
                parts = line.split()
                path = os.path.join(self.root, 'images', parts[0])
                self.path_to_class[path] = parts[1]

        self.classes = list(sorted(set(self.path_to_class.values())))
        self.class_to_idx = {
            label: self.classes.index(label) for label in self.classes
        }

    def __getitem__(self, i):
        sample, _ = super().__getitem__(i)
        path, _ = self.samples[i]
        label = self.path_to_class[path]
        target = self.class_to_idx[label]
        return sample, target

    def __len__(self):
        return super().__len__()


class Imagenet1000(Dataset):
    """ImageNet dataloader"""

    dataset = None

    def __init__(self, root='./data',
            *args, train=True, download=False, **kwargs):
        super().__init__()

        if download:
            self.download(root=root)
        dataset = _Imagenet1000Train if train else _Imagenet1000Val
        self.root = root
        self.dataset = dataset(root, *args, **kwargs)
        self.classes = self.dataset.classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def download(self, root='./'):
        dir = os.path.join(root, 'imagenet-1000')
        dir_train = os.path.join(dir, 'train')
        if os.path.exists(dir) and os.path.exists(dir_train):
            print('==> Already downloaded.')
            return

        msg = ("Please symlink existing ImageNet dataset rather than downloading.")
        raise RuntimeError(msg)

    @staticmethod
    def transform_train(input_size=224):
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size),  # TODO: may need updating
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @staticmethod
    def transform_val(input_size=224):
        return transforms.Compose([
            transforms.Resize(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class _Imagenet1000Train(datasets.ImageFolder):

    def __init__(self, root='./data', *args, **kwargs):
        super().__init__(os.path.join(root, 'imagenet-1000/train'), *args, **kwargs)


class _Imagenet1000Val(datasets.ImageFolder):

    def __init__(self, root='./data', *args, **kwargs):
        super().__init__(os.path.join(root, 'imagenet-1000/val'), *args, **kwargs)

class MiniImagenet(Dataset):
    """ImageNet dataloader"""

    dataset = None

    def __init__(self, root='../mini-imagenet-tools/processed_images',
            *args, train=True, download=False, **kwargs):
        super().__init__()

        dataset = _MiniImagenetTrain if train else _MiniImagenetVal
        self.root = root
        self.dataset = dataset(root, *args, **kwargs)
        # self.classes = self.dataset.classes
        self.classes = []
        classes_path = root.replace('train','').replace('val','') + '/classes.txt'
        print(classes_path)
        with open(classes_path, 'r') as f:
            for i, line in enumerate(f):
                cls = line.strip().split('\t')
                self.classes+=cls
        print(self.classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.train = train

    @staticmethod
    def transform_train(input_size=84):
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size),  # TODO: may need updating
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @staticmethod
    def transform_val(input_size=84):
        return transforms.Compose([
            transforms.Resize(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, i):
        if not self.train:
            return self.dataset[i][0], self.dataset[i][1]+64
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)

class _MiniImagenetTrain(datasets.ImageFolder):

    def __init__(self, root='../mini-imagenet-tools/processed_images', *args, **kwargs):
        super().__init__(os.path.join(root, 'train'), *args, **kwargs)


class _MiniImagenetVal(datasets.ImageFolder):

    def __init__(self, root='../mini-imagenet-tools/processed_images', *args, **kwargs):
        super().__init__(os.path.join(root, 'test'), *args, **kwargs)