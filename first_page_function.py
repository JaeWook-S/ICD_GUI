import gradio as gr
import asyncio
import os
from tkinter import Tk, filedialog
import shutil
import copy
import time

from third_page_function import update_DIC_image
from second_page_function import selected_image_overlay

import model.GFP_ResCNN.test
import model.RFP_ResCNN.test

BASE_DIR = os.path.abspath(os.getcwd())

def show_popup(): return gr.update(visible=True), gr.update(visible=True)
def hide_popup(): return gr.update(visible=False), gr.update(visible=False)
def switch_page(): return gr.update(visible=False), gr.update(visible=True)
def switch_to_third_page(): return gr.update(visible=False), gr.update(visible=True)

def on_browse_folder():
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    folder = filedialog.askdirectory()
    root.destroy()
    if folder:
        return folder  # 선택한 폴더의 절대 경로 반환
    return BASE_DIR  # 선택하지 않을 경우 기본 디렉토리 반환

def list_subfolders(folder):
    # 선택한 폴더가 유효하지 않으면 fallback 디렉토리로 FileExplorer 생성
    if not folder or not os.path.isdir(folder):
        return  gr.FileExplorer(root_dir=BASE_DIR, label="Subfolders", file_count="multiple", elem_id="upload-content")
    try:
        return gr.FileExplorer(root_dir=folder, label="Subfolders", file_count="multiple", elem_id="upload-content")
    except Exception as e:
        print(f"Error updating FileExplorer: {e}")
        return gr.FileExplorer(root_dir=BASE_DIR, label="Subfolders", file_count="multiple", elem_id="upload-content")

def inference(image):
    model.GFP_ResCNN.test.test(image)
    model.RFP_ResCNN.test.test(image)
    
def extract_image_paths(folder_path_dict):
    image_paths = []
    for point in folder_path_dict.values():  # e.g., 'A01'
        for stack in point.values():         # e.g., 'STACK_00000'
            image_paths.extend(stack)        # stack is a list like ['/.../00004.jpg']

    return image_paths

async def progress_bar(folder_path_dict):
    image_paths = extract_image_paths(folder_path_dict)
    total = len(image_paths)

    T = None  # 첫 이미지 처리 시간 저장

    for idx, image_path in enumerate(image_paths, 1):
        if idx == 1:
            t_start = time.perf_counter()

        inference(image_path)

        if idx == 1:
            t_end = time.perf_counter()
            T = t_end - t_start
            total_est_time = T * total

        remaining = total - idx
        est_remaining = T * remaining if T else 0
        est_mins = int(est_remaining) // 60
        est_secs = int(est_remaining) % 60
        est_time_str = f"{total-idx}장 이미지 : {est_mins}분 {est_secs}초 남음"

        percent = int((idx / total) * 100)
        bar = f'<div class="progress-text">Processing: [{"#"*(percent//2)}{" "*(50 - percent//2)}] {percent}%<br>- {est_time_str}</div>'

        yield bar, gr.update(visible=False)
        await asyncio.sleep(0.01)

    progress_done_bar = '<div class="progress-text">Data Inference Success!!<br>You can analysis. </div>'
    yield progress_done_bar, gr.update(visible=True)


    
def cycle_auto_calc(total_hour, total_min, interval_hour, interval_min):
    total_time = total_hour + (total_min / 60)
    interval_time = interval_hour + (interval_min / 60)
    if interval_time == 0:
        return "0 Cycle"
    return str(int(round(total_time / interval_time, 2))) + " Cycle"

def update_progress_with_image_count(uploaded_folder):
    if not uploaded_folder:
        return "<div class='progress-text'>이미지가 업로드되지 않았습니다.</div>"
    if isinstance(uploaded_folder, list):
        count = len(uploaded_folder)
    else:
        count = 1
    return f"<div class='progress-text'>총 이미지(또는 폴더) 개수: {count}</div>"

def extract_first_images(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    well_folders = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])
    
    for well in well_folders:
        bright_path = os.path.join(input_dir, well, "POINT 00001", "BRIGHT")
        if not os.path.isdir(bright_path):
            print(f"[Warning] {bright_path} 폴더가 없습니다.")
            continue
        
        well_output = os.path.join(output_dir, well)
        os.makedirs(well_output, exist_ok=True)
        stack_folders = sorted([
            sf for sf in os.listdir(bright_path)
            if os.path.isdir(os.path.join(bright_path, sf)) and sf.startswith("STACK_")
        ])
        
        for idx, stack_folder in enumerate(stack_folders):
            stack_path = os.path.join(bright_path, stack_folder)
            image_files = sorted([
                f for f in os.listdir(stack_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ])
            
            if not image_files:
                continue
            
            first_image = image_files[0]
            src = os.path.join(stack_path, first_image)
            new_filename = f"STACK{idx:05d}_00001.jpg"
            dst = os.path.join(well_output, new_filename)
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")

def on_gallery_select(selected_value: gr.SelectData, mapping, origin_gallery, overlay_gallery, stack_index, selected_well_toggle_state, selected_idx):
    """
    갤러리에서 이미지를 클릭했을 때 호출되는 콜백.
    'mapping'은 { well_label: well_folder_path } 형태의 딕셔너리라고 가정.
    """
    val = selected_value.value
    idx = selected_value.index
    
    if not isinstance(val, dict):
            # 만약 dict 형태가 아니면(혹은 클릭 정보가 없으면) None 반환
        return None, None, None, None, None, selected_idx, origin_gallery, gr.update(value="All Wells")

    # 갤러리가 전달해 준 딕셔너리에서 caption을 웰 라벨로 사용
    well_label = val.get("caption", "").strip()
    well_folder = mapping.get(well_label) # {stack_00000 : 이미지, stack_00001  ... }
    
    if selected_well_toggle_state:
        idx_update = selected_idx + [well_label] # 선택된 well label 중첩
        

        #print(origin_gallery[idx][0]) # -> path,label tuple return
        image_with_overlay = selected_image_overlay(overlay_gallery[idx][0]) # 선택된 웰을 오버레이
        overlay_gallery[idx] = (image_with_overlay, well_label)
        
        print(idx_update)

        return well_folder, well_label, stack_index, None, None, idx_update, overlay_gallery, gr.update(value="Selected Wells")
    
    else:
        if well_folder is None:
            return None, None, None, None, None, selected_idx, origin_gallery, gr.update(value="All Wells")

        # update_DIC_image( )가 내부에서 "well_folder/POINT00001/BRIGHT/STACK_{index:05d}" 로 접근
        return well_folder, well_label, stack_index, update_DIC_image(well_folder, stack_index), gr.update(value=f"""<div style='font-size: 80px; font-weight: bold; text-align: left; margin: 0 auto; width: 100px'> {well_label} </div>"""), selected_idx, origin_gallery, gr.update(value="All Wells")

