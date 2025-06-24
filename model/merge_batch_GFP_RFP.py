import os
from PIL import Image, ImageEnhance
from concurrent.futures import ThreadPoolExecutor

def resize_to_target(image, target_size):
    """이미지를 target_size로 리사이즈하고 RGBA로 변환"""
    return image.convert("RGBA").resize(target_size, Image.Resampling.LANCZOS)

def enhance_fluorescence(image_path, transparency, enhance_factor=1.5,
                         threshold=50, background_alpha=50, target_size=None):
    """형광 이미지 로드 → 대비 조정 → 알파 채널 강조 → target_size로 리사이즈"""
    image = Image.open(image_path).convert("RGBA")
    image = ImageEnhance.Contrast(image).enhance(enhance_factor)

    datas = image.getdata()
    new_data = []
    for r, g, b, _ in datas:
        if r > threshold or g > threshold or b > threshold:
            new_data.append((r, g, b, transparency))
        else:
            new_data.append((r, g, b, background_alpha))
    image.putdata(new_data)

    if target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)

    return image

def merge_dic_gfp_rfp(dic_path, gfp_path, rfp_path,
                      gfp_transparency=110, rfp_transparency=110,
                      final_contrast_factor=1.5, brightness_factor=0.9):
    """PIL 기반 이미지 병합"""
    dic_image = Image.open(dic_path).convert("RGBA")
    dic_size = dic_image.size

    dic_image = ImageEnhance.Contrast(dic_image).enhance(1)
    dic_image = ImageEnhance.Sharpness(dic_image).enhance(5)

    gfp = enhance_fluorescence(gfp_path, gfp_transparency, target_size=dic_size)
    rfp = enhance_fluorescence(rfp_path, rfp_transparency, target_size=dic_size)

    combined = Image.alpha_composite(dic_image, rfp)
    combined = Image.alpha_composite(combined, gfp)

    combined = ImageEnhance.Contrast(combined).enhance(final_contrast_factor)
    combined = ImageEnhance.Brightness(combined).enhance(brightness_factor)

    return combined

def save_pil_image(args):
    img, path = args
    img.save(path)
    return path

def save_images_parallel(pil_images, save_paths, max_workers=8):
    """PIL 이미지들을 병렬로 저장"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for path in executor.map(save_pil_image, zip(pil_images, save_paths)):
            print(f"[Saved] {path}")

def merge_from_dic_paths(dic_paths,
                         gfp_suffix="_fake_GFP.png",
                         rfp_suffix="_fake_RFP.png",
                         gfp_transparency=110,
                         rfp_transparency=110,
                         final_contrast_factor=1.5,
                         brightness_factor=0.9,
                         merge_suffix="_merge.png",
                         max_workers=8):
    """
    PIL 병합 방식 유지, 저장만 멀티스레드 병렬 처리
    """
    pil_images = []
    save_paths = []

    for dic_path in dic_paths:
        base_dir = os.path.dirname(dic_path)
        base_name = os.path.splitext(os.path.basename(dic_path))[0]
        gfp_path = os.path.join(base_dir, base_name + gfp_suffix)
        rfp_path = os.path.join(base_dir, base_name + rfp_suffix)

        if not os.path.exists(gfp_path) or not os.path.exists(rfp_path):
            print(f"[Skip] Missing: {base_name}")
            continue

        merged = merge_dic_gfp_rfp(dic_path, gfp_path, rfp_path,
                                   gfp_transparency=gfp_transparency,
                                   rfp_transparency=rfp_transparency,
                                   final_contrast_factor=final_contrast_factor,
                                   brightness_factor=brightness_factor)

        save_path = os.path.join(base_dir, base_name + merge_suffix)
        pil_images.append(merged)
        save_paths.append(save_path)

    save_images_parallel(pil_images, save_paths, max_workers=max_workers)
