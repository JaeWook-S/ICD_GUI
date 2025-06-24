import numpy as np
import cv2
import json
import os
from PIL import Image

sc_class_names = ['SC']
dc_class_names = ['DC']
both_class_names = ['SC', 'DC']

def visualize_detections(label_mode, image, json_path, stack_index: int):
    gfp_json_file = os.path.join(json_path, 'best_predictions_GFP.json')
    rfp_json_file = os.path.join(json_path, 'best_predictions_RFP.json')
    what_stack = f"STACK_{str(stack_index).zfill(5)}"

    # PIL → numpy 변환
    if isinstance(image, Image.Image):
        img = np.array(image.convert('RGB'))
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        print("지원되지 않는 이미지 타입입니다.")
        return image

    # 공통 시각화 함수
    def draw_boxes(img, detections, class_names, target_classes):
        for det in detections:
            if det.get("image_id") != what_stack:
                continue

            class_id = det['category_id']
            bbox = det['bbox']
            conf = det['score']

            class_name = class_names[class_id]
            if class_name not in target_classes:
                continue

            label = f"{class_name} {conf:.2f}"
            x, y, w, h = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            if class_name == "SC":
                color = (0, 255, 0)  # Green
            elif class_name == "DC":
                color = (255, 0, 0)  # Red
                
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            label_w, label_h = label_size
            cv2.rectangle(img, (x1, y1 - label_h - 4), (x1 + label_w + 4, y1), (255, 255, 255), -1)
            cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, lineType=cv2.LINE_AA)

    # SC만: GFP에서 SC만 시각화
    if label_mode == "sc" and os.path.exists(gfp_json_file):
        with open(gfp_json_file, 'r') as f:
            detections = json.load(f)
        draw_boxes(img, detections, sc_class_names, ['SC'])

    # DC만: RFP에서 DC만 시각화
    elif label_mode == "dc" and os.path.exists(rfp_json_file):
        with open(rfp_json_file, 'r') as f:
            detections = json.load(f)
        draw_boxes(img, detections, dc_class_names, ['DC'])

    # 둘 다: GFP(SC), RFP(DC) 모두 처리
    elif label_mode == "both":
        if os.path.exists(gfp_json_file):
            with open(gfp_json_file, 'r') as f:
                detections_gfp = json.load(f)
            draw_boxes(img, detections_gfp, sc_class_names, ['SC'])

        if os.path.exists(rfp_json_file):
            with open(rfp_json_file, 'r') as f:
                detections_rfp = json.load(f)
            draw_boxes(img, detections_rfp, dc_class_names, ['DC'])

    return img
