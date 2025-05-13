from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im_batch(input_image, imtype=np.uint8):
    """
    Converts a batch of tensors into list of numpy images.
    Expects input: (B, C, H, W) tensor (range [-1,1] or [0,1])
    Returns: list of (H, W, C) numpy arrays
    """
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.detach().cpu()
    else:
        raise TypeError("Expected input_image to be torch.Tensor")

    image_numpy_batch = []
    for img_tensor in image_tensor:
        img = img_tensor.float().numpy()
        if img.shape[0] == 1:
            img = np.tile(img[0], (3, 1, 1))
        img = (img + 1.0) / 2.0  # [-1,1] → [0,1]
        img = np.clip(img * 255.0, 0, 255)
        img = img.transpose(1, 2, 0)
        image_numpy_batch.append(img.astype(imtype))

    return image_numpy_batch


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
