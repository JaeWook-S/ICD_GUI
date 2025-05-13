import os
import gradio as gr
import pandas as pd
import shutil
import copy 
from PIL import Image

BASE_DIR = os.path.abspath(os.getcwd())


def show_DIC_or_Graph(): return gr.State(value=False), gr.State(value=True)


def is_selected_toggle(state, selected_idx):
    if state == False:
        return True, gr.update(value="Selected Wells"), selected_idx 
    else:
        return False, gr.update(value="All Wells"), []
    
def plate_gallery_visible(plate_type):
    if plate_type == "06 well plate":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    elif plate_type == "12 well plate":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                
    elif plate_type == "24 well plate":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    elif plate_type == "96 well plate":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def all_image_open(well_stack_mapping_path):

    well_stack_mapping_images = {}
    
    for well_name, stack_dict in well_stack_mapping_path.items():
        # stack_dict = {"STACK_00000": [...], "STACK_00001": [...], ...}
        # 대표 이미지는 STACK_00000의 첫 번째 이미지로 가정
        stack_dict_image = {}
        for stack_interval, path in stack_dict.items():
        #print(first_stack_images)
            resized_images = []
            if path: 
            # 있으면 첫 이미지를 썸네일로
                thumb_path = Image.open(path[0])
                thumb_image = thumb_path.resize((64, 64), Image.Resampling.LANCZOS)
                resized_images.append(thumb_image)
            else:
            # 없으면 placeholder
                thumb_path = "https://via.placeholder.com/128?text=No+Image"
                resized_images.append(thumb_path)
                
            stack_dict_image[stack_interval] = resized_images
        well_stack_mapping_images[well_name] = stack_dict_image
        
    return well_stack_mapping_images

def create_2d_mapping(selected_folder, import_selected_folder): # is_import_folder가 true이면 import_selected_folder로 매핑 시작
    """
    root_folder는 사용자가 Browse로 선택한 폴더(= selected_folder)
    그 내부에는 A01 ~ H12 등 96개의 폴더가 있고,
    각 폴더 밑에 POINT 00001/BRIGHT/STACK_00000 ~ STACK_00023 존재한다고 가정.
    """
    mapping = {}
    
    if import_selected_folder != BASE_DIR:
        # 1) 웰 폴더(A01 ~ H12) 찾기
        well_folders = [
            d for d in sorted(os.listdir(import_selected_folder))
            if os.path.isdir(os.path.join(import_selected_folder, d))
        ]
    else:
        # 1) 웰 폴더(A01 ~ H12) 찾기
        well_folders = [
            d for d in sorted(os.listdir(selected_folder))
            if os.path.isdir(os.path.join(selected_folder, d))
        ]

    
    for well_name in well_folders:
        if import_selected_folder != BASE_DIR:
            well_path = os.path.join(import_selected_folder, well_name)
        else:    
            well_path = os.path.join(selected_folder, well_name)
        
        # 2) 스택별 이미지 경로 수집
        stack_dict = {}
        for s in range(24):  # 0 ~ 23
            stack_folder_name = f"STACK_{s:05d}"  # STACK_00000, STACK_00001, ...
            stack_path = os.path.join(well_path, "POINT 00001", "BRIGHT", stack_folder_name)

            # 스택 폴더가 없으면 건너뛰거나 빈 리스트로 처리
            if not os.path.isdir(stack_path):
                stack_dict[stack_folder_name] = []
                continue

            # 3) 이미지 파일 전부 정렬해서 리스트에 담기
            images = sorted([
                os.path.join(stack_path, img)
                for img in os.listdir(stack_path)
                if img.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ])

            stack_dict[stack_folder_name] = images

        # 4) 한 웰에 대한 스택-이미지 정보를 mapping에 저장
        mapping[well_name] = stack_dict
    
    #print(mapping)
    return mapping
    
def update_well_gallery(well_stack_mapping_image, stack_index=0):

    # 2D 매핑 생성
    #well_stack_mapping = create_2d_mapping(selected_folder)

    images_with_labels = []
    for well_name, stack_dict in well_stack_mapping_image.items():
        # stack_dict = {"STACK_00000": [...], "STACK_00001": [...], ...}
        # 대표 이미지는 STACK_00000의 첫 번째 이미지로 가정
        first_stack_images = stack_dict.get(f"STACK_{str(stack_index).zfill(5)}", [])
        #print(first_stack_images)
        if first_stack_images: 
            # 있으면 첫 이미지를 썸네일로
            thumb_path = first_stack_images[0]

        else:
            # 없으면 placeholder
            thumb_path = "https://via.placeholder.com/128?text=No+Image"
        
        images_with_labels.append((thumb_path, well_name))
    
    #focusing_folder_save(well_stack_mapping) # 초점 맞은 폴더 저장
    
    return gr.update(value=images_with_labels),gr.update(value=images_with_labels),gr.update(value=images_with_labels),gr.update(value=images_with_labels), images_with_labels, copy.deepcopy(images_with_labels) # 원본 갤러리 이미지 저장

def page2_export_data_save(all_well_image, stack_index, page2_export_what_well, page2_vis_main, page2_vis_format, page2_graph_main, page2_graph_format):

    if page2_export_what_well == "All Wells": ## 모든 well 저장
        if page2_vis_main == "Visualization Images":
            if page2_graph_main == "Graph": # 모든 well + 이미지 + 그래프 저장
                pass
            else: # 모든 well + 이미지만 저장 
                for well_name, stack_dict in all_well_image.items():
                    present_interval_image = stack_dict.get(f"STACK_{str(stack_index).zfill(5)}", []) # 이 인덱스만 추출
                    present_interval_image_path = present_interval_image[0]

                    # 상위 5단계로 이동 -> 즉 A01~H12폴더가 있는 폴더 안으로 저장할 것
                    export_base_dir = present_interval_image_path
                    for _ in range(5):
                        export_base_dir = os.path.dirname(export_base_dir)

                    save_path = os.path.join(export_base_dir, "exported_data")

                    os.makedirs(save_path, exist_ok=True)

                    file_name = f"{well_name}_STACK_{str(stack_index).zfill(5)}.{page2_vis_format.lower()}"
                    save_path = os.path.join(save_path, file_name)

                    # 이미지 복사
                    shutil.copy(present_interval_image_path, save_path)
                
        else: # 모든 well + 이미지 X 
            if page2_graph_main == "Graph": # 모든 well + 이미지 X + 그래프 저장
                pass
            else: # 모든 well + 이미지 X + 그래프 X -> 잘못 눌렸다고 판단하여 아무 동작도 안함
                pass
                
    else: # selected well 저장
        pass
    
    return "All Wells", None, None, None, None # 저장을 누르면 선택된 것 초기화

def focusing_folder_save(well_stack_mapping):
    
    for well_name, stack_dict in well_stack_mapping.items():
        #print(stack_dict)
        for stack, DIC_image_path in stack_dict.items():
            DIC_image_path = DIC_image_path[0]
            
            export_base_dir = DIC_image_path
            for i in range(5):
                export_base_dir = os.path.dirname(export_base_dir)
                
                if i == 0:
                    present_interval_state = os.path.basename(export_base_dir) # STACK_00000~00023 추출
                elif i == 1:
                    bright_dir = os.path.basename(export_base_dir) # BRIGHT 추출
                elif i == 3:
                    well_dir = os.path.basename(export_base_dir) # A01~H12 추출
            
            _, interval = present_interval_state.split('_')    
            save_path = os.path.join(export_base_dir, "exported_data", well_dir, bright_dir)
            
            os.makedirs(save_path, exist_ok=True)

            save_path = os.path.join(save_path, interval) # 이미지명 변경 00000~ 00023으로 
            shutil.copy(DIC_image_path, save_path)
            
def selected_image_overlay(selected_image):
    img = selected_image.convert("RGBA")

    # 같은 크기의 주황색 오버레이 이미지 생성
    orange_overlay = Image.new("RGBA", img.size, (255, 165, 0, 160))  # 주황 + 투명도 조절

    # 이미지 합성
    combined = Image.alpha_composite(img, orange_overlay)

    return combined.convert("RGB")  # RGB로 변환해 다시 반환

def overlay_image_init(origin):
    overlay_init = copy.deepcopy(origin)
    return overlay_init, overlay_init, overlay_init, overlay_init, overlay_init