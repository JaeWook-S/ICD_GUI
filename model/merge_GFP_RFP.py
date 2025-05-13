from PIL import Image, ImageEnhance

def resize_to_target(image, target_size):
    """이미지를 target_size로 리사이즈하고 RGBA로 변환"""
    return image.convert("RGBA").resize(target_size, Image.Resampling.LANCZOS)

def enhance_fluorescence(image_path, transparency, enhance_factor=1.5, threshold=50, background_alpha=50, target_size=None):
    """형광 이미지 로드 → 대비 조정 → 알파 채널 강조 → target_size로 리사이즈"""
    image = Image.open(image_path).convert("RGBA")
    enhancer = ImageEnhance.Contrast(image)
    image_enhanced = enhancer.enhance(enhance_factor)

    # 알파 채널 수정
    datas = image_enhanced.getdata()
    new_data = []
    for item in datas:
        if item[0] > threshold or item[1] > threshold or item[2] > threshold:
            new_data.append((item[0], item[1], item[2], transparency))
        else:
            new_data.append((item[0], item[1], item[2], background_alpha))
    image_enhanced.putdata(new_data)

    # DIC 사이즈 기준으로 리사이즈
    if target_size:
        image_enhanced = image_enhanced.resize(target_size, Image.Resampling.LANCZOS)

    return image_enhanced

def merge_dic_gfp_rfp(dic_path, gfp_path, rfp_path,
                      gfp_transparency=110, rfp_transparency=110,
                      final_contrast_factor=1.5, brightness_factor=0.9):
    """
    DIC, GFP, RFP 이미지 병합
    - DIC 사이즈에 맞게 GFP/RFP 리사이즈
    - 알파 합성 방식으로 시각화
    - 최종 대비 및 밝기 조절
    """
    # DIC 처리
    dic_image = Image.open(dic_path).convert("RGBA")
    dic_size = dic_image.size  # 기준 사이즈

    # DIC 대비 및 샤프닝
    dic_image = ImageEnhance.Contrast(dic_image).enhance(1)
    dic_image = ImageEnhance.Sharpness(dic_image).enhance(5)

    # GFP & RFP 강화 및 리사이즈
    gfp_enhanced = enhance_fluorescence(gfp_path, transparency=gfp_transparency, target_size=dic_size)
    rfp_enhanced = enhance_fluorescence(rfp_path, transparency=rfp_transparency, target_size=dic_size)

    # 병합: DIC + RFP + GFP
    combined = Image.alpha_composite(dic_image, rfp_enhanced)
    combined = Image.alpha_composite(combined, gfp_enhanced)

    # 최종 대비 + 밝기 조정
    combined = ImageEnhance.Contrast(combined).enhance(final_contrast_factor)
    combined = ImageEnhance.Brightness(combined).enhance(brightness_factor)

    return combined  # PIL.Image 객체 반환
