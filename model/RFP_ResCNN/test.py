import os
import torch
from model.RFP_ResCNN.options.test_options import TestOptions
from model.RFP_ResCNN.data import CreateDataLoader
from model.RFP_ResCNN.models import create_model
from model.RFP_ResCNN.util.visualizer import Visualizer
from PIL import Image
import numpy as np
from model.RFP_ResCNN.data.base_dataset import get_transform

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


def test(batch_image_paths):

    opt = TestOptions().parse()
    opt.nThreads = 1
    opt.batchSize = len(batch_image_paths)  # e.g., 10
    opt.serial_batches = True
    opt.no_flip = True

    model = create_model(opt)

    # 배치 이미지 불러오기
    input_batch = load_batch_images_as_dict(batch_image_paths, opt)
    model.set_input(input_batch)
    model.test()
    visuals = model.get_current_visuals()
    img_paths = input_batch['A_paths']

    #print(f'Processing batch: {img_paths}')

    # 저장
    for i, fake_B in enumerate(visuals['fake_B']):
        if isinstance(fake_B, torch.Tensor):
            fake_B = fake_B.detach().cpu().numpy()

        # 정규화 여부에 따라 scale
        if fake_B.max() <= 1.0:
            fake_B = (fake_B * 255)

        fake_B = fake_B.astype(np.uint8)

        # fake_B shape 확인
        if fake_B.ndim == 3 and fake_B.shape[0] in [1, 3]:  # (C, H, W)
            fake_B = fake_B.transpose(1, 2, 0)
        elif fake_B.ndim == 2:  # (H, W)
            pass
        elif fake_B.ndim == 3 and fake_B.shape[2] in [1, 3]:  # already (H, W, C)
            pass
        else:
            raise ValueError(f"Unexpected fake_B shape: {fake_B.shape}")

        save_dir = os.path.dirname(img_paths[i])
        save_name = os.path.splitext(os.path.basename(img_paths[i]))[0] + '_fake_RFP.png'
        save_path = os.path.join(save_dir, save_name)
        fake_B_image = Image.fromarray(fake_B)
        fake_B_image.save(save_path)
