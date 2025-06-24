import gradio as gr
import pandas as pd
import os
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from model.visualization_detection import visualize_detections

def show_popup(): return gr.update(visible=True), gr.update(visible=True)
def hide_popup(): return gr.update(visible=False), gr.update(visible=False)
def slider_position_change(index): return gr.update(value=index+1)


def update_slider(cycle_value):
    if isinstance(cycle_value, str):
        try:
            # 실수 문자열 처리
            cycle_value = int(cycle_value.split()[0])
        except:
            cycle_value = 24  # fallback
    return gr.Slider(minimum=1, maximum=cycle_value, value=1, step=1, interactive=False)

#############################
# (1) STACK 이미지 로드 함수
#############################
def get_stack_image_path(well_folder, stack_index):
    """
    well_folder: 예) .../IMAGES/A01
    stack_index: 0~23 (STACK_00001 ~ STACK_00024)
    """
    #print("well folder:", well_folder)
    folder_name = f"STACK_{stack_index+1:05d}"  # ex) stack_index=0 -> STACK_00001
    #print("folder name:", folder_name)

    bright_path = os.path.join(well_folder, "POINT 00001", "BRIGHT", folder_name)
    #print("bright_path:", bright_path)
    if not os.path.isdir(bright_path):
        return None

    # 해당 STACK 폴더 내 첫 번째 이미지(예: 00001.png)
    image_files = sorted([
        f for f in os.listdir(bright_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])
    if not image_files:
        return None
    return os.path.join(bright_path, image_files[0])

#############################
# (2) DIC, Graph, DataFrame 업데이트 콜백
#############################
def update_peak_table(sc_peak_count, sc_peak_time, dc_peak_count, dc_peak_time):

    data = {
        "": ["SC", "DC"],
        "Peak Count": [sc_peak_count, dc_peak_count],
        "Peak Time": [sc_peak_time, dc_peak_time]
    }

    return pd.DataFrame(data)

def update_DIC_image(pseudo_color_toggle, sc_toggle, dc_toggle, well_data, stack_index: int):
    """
    well_data: 
    {
      "STACK_00000": [...이미지 경로들...],
      ...
    }
    """
    image_list = well_data.get(f"STACK_{str(stack_index).zfill(5)}", [])

    def placeholder():
        return "https://via.placeholder.com/128?text=No+Image"

    def load_dic_image():
        if not image_list:
            return placeholder()
        return Image.open(image_list[0]).convert("RGB")

    # Determine label mode
    if sc_toggle and dc_toggle:
        label_mode = "both"
    elif sc_toggle:
        label_mode = "sc"
    elif dc_toggle:
        label_mode = "dc"
    else:
        label_mode = None

    # Choose base image
    if pseudo_color_toggle:
        base_image = show_merge_GFP_RFP(well_data, stack_index)
    else:
        if not image_list:
            return placeholder()
        base_image = load_dic_image()

    # Add labels if needed
    if label_mode:
        return show_cell_label(label_mode, base_image, well_data, stack_index)
    
    return base_image if pseudo_color_toggle else image_list[0]


    
def count_stack_by_class(detections, stack_len):
    stack_keys = [f"STACK_{str(i).zfill(5)}" for i in range(stack_len)]
    return [sum(1 for d in detections if d["image_id"] == key) for key in stack_keys]


def update_Graph_and_Peak(well_data, cycle, stack_idx: int):
    cycle = int(cycle.split()[0])
    time = np.arange(0, int(cycle), 1)

    # JSON 경로 추출
    first_key = next(iter(well_data))
    first_path = well_data[first_key][0]
    json_path = os.path.dirname(os.path.dirname(first_path))

    stack_len = len(well_data)

    # SC 예측 수
    try:
        with open(os.path.join(json_path, 'best_predictions_GFP.json'), 'r') as f:
            detections_sc = json.load(f)
        sc = np.array(count_stack_by_class(detections_sc, stack_len))
    except FileNotFoundError:
        print("[INFO] GFP json 파일 없음 → SC 전부 0으로 처리")
        sc = np.zeros(stack_len)

    # DC 예측 수
    try:
        with open(os.path.join(json_path, 'best_predictions_RFP.json'), 'r') as f:
            detections_dc = json.load(f)
        dc = np.array(count_stack_by_class(detections_dc, stack_len))
    except FileNotFoundError:
        print("[INFO] RFP json 파일 없음 → DC 전부 0으로 처리")
        dc = np.zeros(stack_len)

    # peak 정보
    sc_peak_count = f"{sc.max()} Cell"
    sc_peak_time = f"{sc.argmax()} Cycle"
    dc_peak_count = f"{dc.max()} Cell"
    dc_peak_time = f"{dc.argmax()} Cycle"
    peak_data = update_peak_table(sc_peak_count, sc_peak_time, dc_peak_count, dc_peak_time)

    # 시각화
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(time, sc, color='green', label='pred_S.C', marker='o', linewidth=5)
    ax.plot(time, dc, color='red', label='pred_D.C', marker='s', linewidth=5)

    min_y, max_y = min(sc[stack_idx], dc[stack_idx]), max(sc[stack_idx], dc[stack_idx])
    ax.vlines(stack_idx, min_y, max_y, colors='black', linestyles='dashed', linewidth=5)
    ax.plot(stack_idx, min_y, 'o', color='white', markersize=10, markeredgecolor='black', zorder=5)
    ax.plot(stack_idx, max_y, 'o', color='white', markersize=10, markeredgecolor='black', zorder=5)

    ax.set_xticks(time)
    ax.legend(fontsize=16)
    ax.grid(False)
    plt.tight_layout()
    plt.close(fig)

    return fig, peak_data


def show_merge_GFP_RFP(well_data, stack_index: int):

    image_list = well_data.get(f"STACK_{str(stack_index).zfill(5)}", [])
    
    if image_list:
        path = image_list[3]
        image = Image.open(path).convert("RGB")
        return image
        
    else:
        return "https://via.placeholder.com/128?text=No+Image"

    
def show_cell_label(label_mode, current_image, well_data, stack_index: int):
    
    first_key = next(iter(well_data))
    first_path = well_data[first_key][0] # well_data dictionary에서 맨 앞 경로 추출
    
    json_path = os.path.dirname(os.path.dirname(first_path)) # json이 있는 경로
    
    return visualize_detections(label_mode, current_image, json_path, stack_index)
 

#############################
# (3) stack_index 조절 함수
#############################
def prev_stack(current_index): new_index = current_index - 1 ; return max(new_index, 0)     
def next_stack(current_index, cycle_value): 
    new_index = current_index + 1 
    
    if isinstance(cycle_value, str):
        try:
            # 실수 문자열 처리
            cycle_value = int(cycle_value.split()[0])
        except:
            cycle_value = 24  # fallback
            
    return min(new_index, cycle_value-1)

def page3_export_data_save(current_image, current_graph, stack_index, well_label, all_interval_folder_path, page3_vis_main, page3_vis_format, page3_graph_main, page3_graph_format): 
    # dict로 모든 interval 경로가 저장 -> 1개만 뽑아서 경로 설정할거임
    first_key = list(all_interval_folder_path.keys())[0]
    folder_path = all_interval_folder_path[first_key][0]


    if page3_vis_main == "Visualization Images":
        if page3_graph_main == "Graph": # 이미지 + 그래프 저장 -- > 구현 필요
            pass
        
        else: # 이미지만 저장 
            for _ in range(5): # A01~H12가 있는 폴더 위치에 저장할 것 
                folder_path = os.path.dirname(folder_path)
                
            # 저장 경로 설정 및 폴더 생성
            export_folder = os.path.join(folder_path, "exported_data_with_model_inference")
            os.makedirs(export_folder, exist_ok=True)
            
            file_name = f"{well_label}_STACK_{str(stack_index).zfill(5)}.{page3_vis_format.lower()}"
            save_path = os.path.join(export_folder, file_name) # 최종 저장 경로
            
            if isinstance(current_image, np.ndarray):
                image = Image.fromarray(current_image)
            else:
                image = current_image  # 이미 PIL이면 그대로

            image.save(save_path)
            
            print("페이지 3 이미지 추출 완료")
            
    else: # 이미지 X -> 구현 필요
        if page3_graph_main == "Graph": # 이미지 X + 그래프 저장
            pass
        else: # 이미지 X + 그래프 X -> 잘못 눌렸다고 판단하여 아무 동작도 안함
            pass

    
    return None, None, None, None # 저장을 누르면 선택된 것 초기화