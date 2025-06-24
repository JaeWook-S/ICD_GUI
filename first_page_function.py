import torch
import gradio as gr
import asyncio
import os
from tkinter import Tk, filedialog
import shutil
import time

from third_page_function import update_DIC_image
from second_page_function import selected_image_overlay, all_image_open

from model.GFP_ResCNN.options.test_options import GFP_TestOptions
from model.GFP_ResCNN.models import GFP_create_model
from model.RFP_ResCNN.options.test_options import RFP_TestOptions
from model.RFP_ResCNN.models import RFP_create_model

import model.GFP_ResCNN.test
import model.RFP_ResCNN.test
from model.merge_batch_GFP_RFP import merge_from_dic_paths
from model.cell_detection_part.SC.inference_SC import run_sc_inference
from model.cell_detection_part.DC.inference_DC import run_dc_inference


BASE_DIR = os.path.abspath(os.getcwd())

def model_load(what_model):
    if what_model == "GFP":
        gfp_opt = GFP_TestOptions().parse()
        gfp_model = GFP_create_model(gfp_opt)
        return gfp_model
    elif what_model == "RFP":
        rfp_opt = RFP_TestOptions().parse()
        rfp_model = RFP_create_model(rfp_opt)
        return rfp_model

gfp_model = model_load("GFP")
rfp_model = model_load("RFP")

def what_folder_type(): return gr.State(value=True), gr.State(value=False)
def hide_popup(): return gr.update(visible=False), gr.update(visible=False)
def switch_page(): return gr.update(visible=False), gr.update(visible=True)
def switch_to_third_page(): return gr.update(visible=False), gr.update(visible=True)

def show_popup(information_select_complete):
    if information_select_complete: 
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)

# 폴더 처리 함수
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

# init or import 폴더가 바뀌게 되면 나머지 한개는 초기화
def reset_init_or_import_folder(): return BASE_DIR, gr.FileExplorer(root_dir=BASE_DIR, show_label=False, file_count="multiple", elem_id="upload-content")

# 정보들이 모두 입력됐는지 확인  
def check_information_val(information_plate_type, information_total_time_hour, information_interval_hour, analysis_protocol_SC, analysis_protocol_DC):
    if information_plate_type == None or information_total_time_hour == 0 or information_interval_hour == 0 or analysis_protocol_SC == None or analysis_protocol_DC == None:
        return False
    else:
        return True
    
# 모델 inference 및 진행바 함수
def inference(image, analysis_protocal_SC, analysis_protocal_DC):
     #gfp inference
    gfp_model.netG.cuda()
    model.GFP_ResCNN.test.test(gfp_model, image)
    gfp_model.netG.cpu()
    torch.cuda.empty_cache()

     #RFP inference
    rfp_model.netG.cuda()
    model.RFP_ResCNN.test.test(rfp_model, image)
    rfp_model.netG.cpu()
    torch.cuda.empty_cache()

     #DIC + GFP + RFP 병합
    merge_from_dic_paths(image)
    run_sc_inference(image)
    run_dc_inference(image)
    #if analysis_protocal_SC == "ON" and analysis_protocal_DC == "OFF":
    #   # sc만 infernece 
    #   run_sc_inference(image)
    #elif analysis_protocal_SC == "OFF" and analysis_protocal_DC == "ON":
    #   # dc inference
    #   run_dc_inference(image)
    #elif analysis_protocal_SC == "ON" and analysis_protocal_DC == "ON":
    #   run_sc_inference(image)
    #   run_dc_inference(image)
    
def extract_image_paths(folder_path_dict):
    image_paths = []
    for point in folder_path_dict.values():  # e.g., 'A01'
        for stack in point.values():         # e.g., 'STACK_00000'
            image_paths.extend(stack)        # stack is a list like ['/.../00004.jpg']

    return image_paths

async def inference_progress_bar(folder_path_dict, import_selected_folder, analysis_protocal_SC, analysis_protocal_DC, cycle_auto_calc_text):
    image_paths = extract_image_paths(folder_path_dict)
    total = len(image_paths)
    batch_size = int(cycle_auto_calc_text.split()[0]) # 배치 사이즈를 한 사이클로 진행

    num_batches = (total + batch_size - 1) // batch_size  # ceil

    T = None  # 첫 배치 처리 시간 저장
    if import_selected_folder == BASE_DIR: # 초기 폴더일 때만 inference 진행
        for batch_idx in range(num_batches):
            batch = image_paths[batch_idx*batch_size : (batch_idx+1)*batch_size]

            if batch_idx == 0:
                t_start = time.perf_counter()

            inference(batch, analysis_protocal_SC, analysis_protocal_DC)

            if batch_idx == 0:
                t_end = time.perf_counter()
                T = t_end - t_start
                total_est_time = T * num_batches

            remaining_batches = num_batches - batch_idx - 1
            est_remaining = T * remaining_batches if T else 0
            est_mins = int(est_remaining) // 60
            est_secs = int(est_remaining) % 60
            est_time_str = f"{total - (batch_idx+1)*batch_size}장 이미지 : {est_mins}분 {est_secs}초 남음"

            percent = int(((batch_idx+1) * batch_size / total) * 100)
            bar = f'<div class="progress-text">Processing: [{"#"*(percent//2)}{" "*(50 - percent//2)}] {percent}%<br>- {est_time_str}</div>'

            yield bar, gr.update(visible=False), None
            await asyncio.sleep(0.01)
    
    progress_bar_for_thumbnail = '<div class="progress-text">Processing for Program !!<br>This may take about a minute. </div>'
    yield progress_bar_for_thumbnail, gr.update(visible=False), None
    await asyncio.sleep(0.1)
    
    well_stack_mapping_image = all_image_open(folder_path_dict) # 썸네일 이미지를 위해 이미지 사이즈를 작게 만드는 중
    
    progress_done_bar = '<div class="progress-text">Data Inference Success!!<br>You can analysis. </div>'
    yield progress_done_bar, gr.update(visible=True), well_stack_mapping_image

# cycle 자동 계산 함수
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

def on_gallery_select(selected_value: gr.SelectData, mapping, origin_gallery, overlay_gallery, stack_index, selected_well_toggle_state, selected_idx, pseudo_color_toggle, SC_toggle, DC_toggle):
    """
    갤러리에서 이미지를 클릭했을 때 호출되는 콜백.
    'mapping'은 { well_label: well_folder_path } 형태의 딕셔너리라고 가정.
    """
    val = selected_value.value
    idx = selected_value.index
    
    if not isinstance(val, dict):
            # 만약 dict 형태가 아니면(혹은 클릭 정보가 없으면) None 반환
        return None, None, None, None, None, selected_idx, gr.update(value=origin_gallery), gr.update(value="All Wells")

    # 갤러리가 전달해 준 딕셔너리에서 caption을 웰 라벨로 사용
    well_label = val.get("caption", "").strip()
    well_folder = mapping.get(well_label) # {stack_00000 : 이미지, stack_00001  ... }
    
    if selected_well_toggle_state:
        idx_update = selected_idx + [well_label] # 선택된 well label 중첩
        

        #print(origin_gallery[idx][0]) # -> path,label tuple return
        image_with_overlay = selected_image_overlay(overlay_gallery[idx][0]) # 선택된 웰을 오버레이
        overlay_gallery[idx] = (image_with_overlay, well_label)
        
        print(idx_update)

        return well_folder, well_label, stack_index, None, None, idx_update, gr.update(value=overlay_gallery), gr.update(value="Selected Wells")
    
    else:
        if well_folder is None:
            return None, None, None, None, None, selected_idx, origin_gallery, gr.update(value="All Wells")

        # update_DIC_image( )가 내부에서 "well_folder/POINT00001/BRIGHT/STACK_{index:05d}" 로 접근
        return well_folder, well_label, stack_index, update_DIC_image(pseudo_color_toggle, SC_toggle, DC_toggle, well_folder, stack_index), gr.update(value=f"""<div style='font-size: 80px; font-weight: bold; text-align: left; margin: 0 auto; width: 100px'> {well_label} </div>"""), selected_idx, gr.update(value=origin_gallery), gr.update(value="All Wells")

