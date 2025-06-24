import os
import torch
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from model.GFP_ResCNN.options.test_options import GFP_TestOptions
from model.GFP_ResCNN.models import GFP_create_model
from model.GFP_ResCNN.data.base_dataset import get_transform
from collections import OrderedDict


def load_batch_images_as_dict(image_paths, opt):
    transform = get_transform(opt)
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img)

        if opt.input_nc == 1:
            tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = tmp.unsqueeze(0)
        
        images.append(img)

    batch_tensor = torch.stack(images, dim=0)  # (B, C, H, W)
    return {'A': batch_tensor, 'A_paths': image_paths}


def save_image_numpy(args):
    img_np, path = args

    # float32 → uint8
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)

    # (C, H, W) → (H, W, C)
    if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
        img_np = img_np.transpose(1, 2, 0)

    img_pil = Image.fromarray(img_np)
    img_pil.save(path)
    return path


def save_fake_B_images_parallel(fake_B_list, img_paths, suffix="_fake_GFP.png", max_workers=8):
    jobs = []
    for i, fake_B in enumerate(fake_B_list):
        base = os.path.splitext(os.path.basename(img_paths[i]))[0]
        save_path = os.path.join(os.path.dirname(img_paths[i]), base + suffix)
        jobs.append((fake_B, save_path))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(save_image_numpy, jobs):
            print(f"[Saved] {result}")


def test(model, batch_image_paths):
    
    opt = GFP_TestOptions().parse()
    opt.nThreads = 1
    opt.batchSize = len(batch_image_paths)
    opt.serial_batches = True
    opt.no_flip = True

    # 입력 이미지 배치 로딩
    input_batch = load_batch_images_as_dict(batch_image_paths, opt)
    model.set_input(input_batch)
    model.test()

    visuals = model.get_current_visuals()  # OrderedDict with 'fake_B' as np arrays
    fake_B_list = visuals['fake_B']
    save_fake_B_images_parallel(fake_B_list, input_batch['A_paths'])
