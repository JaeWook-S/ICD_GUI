import gradio as gr
import pandas as pd
import os
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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
# (2) DIC, Graph 이미지 업데이트 콜백
#############################
def update_DIC_image(well_data, stack_index: int):
    """
    well_data는 예: 
    {
      "STACK_00000": [...이미지 경로들...],
      "STACK_00001": [...],
      ...
    }
    """
    #print(stack_index)
    # 스택 폴더 이름
    image_list = well_data.get(f"STACK_{str(stack_index).zfill(5)}", [])

    if image_list: 
            # 있으면 첫 이미지를 썸네일로
        return image_list[0]
    else:
            # 없으면 placeholder
        return "https://via.placeholder.com/128?text=No+Image"
    
def update_ICD_Graph(well_label, cycle, stack_idx: int):
    # well label에 맞는 그래프 가져오기  -> 구현 필요 // 임시로 temp_graph넣엇음
    # stack_idx는 현재 어느 위치인지 표시를 위해 // cycle은 현재 몇번째 interval인지 표시를 위해 (ex. cycle = '24 Cycle')
    cycle = int(cycle.split()[0])
    time = np.arange(0, int(cycle), 1) 
    
    # 예시 데이터 -> 여기에 모델 예측한 값 넣으면 됨
    sc = np.array([10, 12, 15, 20, 25, 30, 40, 55, 60, 70, 50, 30, 20, 10, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2])
    dc = np.array([15, 18, 20, 25, 30, 40, 60, 100, 150, 200, 300, 310, 320, 310, 300, 290, 295, 300, 305, 300, 290, 280, 275, 270])
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    ax.plot(time, sc, color='green', label='pred_S.C', marker='o', linewidth=5)
    ax.plot(time, dc, color='red', label='pred_D.C', marker='s', linewidth=5)

    # 수직선 추가: stack_idx에서 더 높은 값을 기준으로 선 높이 결정
    min_y, max_y = min(sc[stack_idx], dc[stack_idx]), max(sc[stack_idx], dc[stack_idx])
    ax.vlines(stack_idx, min_y, max_y, colors='black', linestyles='dashed', linewidth=5) # 현재 interval의 선 표시
    ax.plot(stack_idx, min_y, 'o', color='white', markersize=10, markeredgecolor='black', zorder=5) # 현재 interval의 그래프 위의 SC,DC 하얀 동그라미 표시
    ax.plot(stack_idx, max_y, 'o', color='white', markersize=10, markeredgecolor='black', zorder=5)
    
    ax.set_xticks(time) # x축 눈금 0~cycle까지 설정
    ax.legend(fontsize=16)# 범례 및 레이아웃
    ax.grid(False)
    plt.tight_layout()

    plt.close(fig)
    return fig

    
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