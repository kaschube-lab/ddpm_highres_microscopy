import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import random
from PIL import ImageOps, ImageFilter

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', 
    '.BMP', '.tif',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        
        images = np.unique([os.path.basename(path)for path in images])

    return images

def pil_loader(path):
    return Image.open(path).convert('L')

def get_min_max():
    min_high, max_high = -1, 1

    return (min_high, max_high)

def denorm_tensor():
    mean_high, std_high = 0.5, 0.5

    return transforms.Normalize(-(mean_high/std_high), (1/std_high))

def denorm_np():
    mean_high, std_high = 0.5, 0.5
    return lambda x: x*std_high + mean_high

class MicroscopyDataset(data.Dataset):
    def __init__(self, data_root, data_flist, type,
                 data_len=-1, image_size=[256, 256],
                 phase='train',loader=pil_loader,
                 norm=True):
        self.data_root = data_root
        self.phase = phase
        self.type = type
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist

        self.mean_low, self.std_low = 0.5, 0.5
        self.mean_high, self.std_high = 0.5, 0.5

        self.tfs_low = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.mean_low], std=[self.std_low]),
        ])
        self.tfs_high = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.mean_high], std=[self.std_high]),
        ])

        self.loader = loader
        self.image_size = image_size
        self.norm = norm

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5)

        if self.type in ['synapse', 'microtubule']:
            img = self.loader('{}/{}/{}'.format(self.data_root, 'high_registered', file_name))
            cond_image = self.loader('{}/{}/{}'.format(self.data_root, 'low_padded', file_name))
        else:
            img = self.loader('{}/{}/{}'.format(self.data_root, 'high', file_name))
            cond_image = self.loader('{}/{}/{}'.format(self.data_root, 'low', file_name))

        if self.phase == 'train':
            self.aug(cond_image, img)

        img = self.tfs_high(img)
        cond_image = self.tfs_low(cond_image)

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)
    
    def aug(self, img, gt):
        if random.random() < 0.5:
            img = ImageOps.flip(img)
            gt = ImageOps.flip(gt)
        if random.random() < 0.5:
            img = img.rotate(90)
            gt = gt.rotate(90)
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=3))
        return img, gt


