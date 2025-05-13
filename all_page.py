import gradio as gr
import asyncio
import os
from tkinter import Tk, filedialog
import shutil

from gradio_toggle import Toggle
import pandas as pd

from first_page_function import on_browse_folder, on_gallery_select, list_subfolders, progress_bar, cycle_auto_calc, show_popup, hide_popup, update_progress_with_image_count, extract_first_images, inference, what_folder_type
from second_page_function import update_well_gallery, show_DIC_or_Graph, create_2d_mapping, page2_export_data_save, focusing_folder_save, plate_gallery_visible, is_selected_toggle, all_image_open, overlay_image_init
from third_page_function import update_DIC_image, prev_stack, next_stack, get_stack_image_path, slider_position_change, update_slider, page3_export_data_save, update_Graph_and_Peak, merge_GFP_RFP

BASE_DIR = os.path.abspath(os.getcwd())
    
peak_data = {
    "": ["SC", "DC"],
    "Peak Count": ["", ""],
    "Peak Time": ["", ""]
}

### fuction ###
def switch_page(selected_well_toggle_state=False): 
    if selected_well_toggle_state == True:
        return gr.update(visible=False), gr.update(visible=True)
    return gr.update(visible=True), gr.update(visible=False)


def main_page():
    # 2D 매핑(웰×STACK)
    well_stack_mapping_path= gr.State({})
    well_stack_mapping_image = gr.State({})
    
    # 기본 선택 폴더를 BASE_DIR로 설정
    selected_folder = gr.State(value=BASE_DIR)
    import_selected_folder = gr.State(value=BASE_DIR)
    
######################### 
#######  page 1 #########
#########################

    with gr.Column(visible=True) as page1:
        with gr.Row(): # 첫 화면 Markdown다
            with gr.Column(scale=1):
                gr.Markdown("# Analysis Files")
            with gr.Column(elem_classes="center-align", scale=1):
                gr.Markdown("# Import Files")
            with gr.Column(elem_classes="center-align", scale=2):
                gr.Markdown("# Information")
            
        with gr.Column():
            with gr.Row():# 업로더 및 information
                ######################################################## folder uploader ############################################################
                
                with gr.Column(scale=1):
                    init_browse_button = gr.Button("Browse", elem_id="analysis-browse-button")
                    # 초기 FileExplorer 생성 (출력 대상은 init_upload_folder)
                    init_upload_folder = gr.FileExplorer(root_dir=BASE_DIR, show_label=False, file_count="multiple", elem_id="upload-content")
                    # Browse 버튼 클릭 시 선택 폴더를 업데이트
                    init_browse_button.click(fn=on_browse_folder, inputs=[], outputs=selected_folder)
                    # 선택된 폴더 변경 시 FileExplorer 자체를 새 인스턴스로 반환하여 교체
                    selected_folder.change(fn=list_subfolders, inputs=selected_folder, outputs=[init_upload_folder])
                
                with gr.Column(scale=1):
                    import_browse_button = gr.Button("Browse", elem_id="analysis-browse-button")
                    import_upload_folder = gr.FileExplorer(root_dir=BASE_DIR, show_label=False, file_count="multiple", elem_id="upload-content")
                    import_browse_button.click(fn=on_browse_folder, inputs=[], outputs=import_selected_folder)
                    # 선택된 폴더 변경 시 FileExplorer 자체를 새 인스턴스로 반환하여 교체
                    import_selected_folder.change(fn=list_subfolders, inputs=import_selected_folder, outputs=[import_upload_folder])
                
                ####################################################### information part #############################################################
                with gr.Column(scale=2):
                    with gr.Column(elem_id="information-content"):
                        with gr.Row(elem_id="information-content"):
                            with gr.Row(scale=1):
                                gr.Text("\nPlate Type\n", show_label=False)
                            with gr.Row(scale=2):
                                information_plate_type = gr.Radio(["06 well plate", "12 well plate", "24 well plate", "96 well plate"], label=None, show_label=False, elem_id="radio-plate-type")
                        
                        with gr.Row(elem_id="information-content"):
                            with gr.Row(scale=1):
                                gr.Text("\nTotal time\n", show_label=False)
                            with gr.Row(scale=2):
                                information_total_time_hour = gr.Number(value=0, minimum=0, step=1, label="Hour", interactive=True)
                                information_total_time_min = gr.Number(value=0, minimum=0, maximum=59, step=5, label="Min", interactive=True)
                            
                        with gr.Row(elem_id="information-content"):
                            with gr.Row(scale=1):
                                gr.Text("\nInterval\n", show_label=False)
                            with gr.Row(scale=2):
                                information_interval_hour = gr.Number(value=0, minimum=0, step=1, label="Hour", interactive=True)
                                information_interval_min = gr.Number(value=0, minimum=0, maximum=59, step=5, label="Min", interactive=True)
                            
                        with gr.Row(elem_id="information-content"):
                            with gr.Row(scale=1):
                                gr.Text("Cycle", show_label=False)
                            with gr.Row(scale=2):
                                cycle_auto_calc_text = gr.Text("0 Cycle", show_label=False)
                            
            ####################################################### information part #############################################################   
            with gr.Row(): # process바 및 SC,DC 선택
                with gr.Row(): 
                    with gr.Column(scale=1):               
                        gr.Markdown("# Progress")
                        model_processing_time = gr.HTML("<div class='progress-text'>Hello World!</div>")    
                
                    with gr.Column():
                        gr.Markdown(value="\n# Analysis Protocol", elem_id="center-align")
                        with gr.Row():
                            with gr.Column(elem_id="information-content"):
                                with gr.Row(elem_id="information-content"):
                                    with gr.Row(scale=1):
                                        gr.Text("SC(Swollen Cell)", show_label=False)
                                    with gr.Row(scale=2):
                                        analysis_protocol_SC = gr.Radio(
                                        choices=["ON", "OFF"], label=None, show_label=False
                                    )
                                with gr.Row(elem_id="information-content"):
                                    with gr.Row(scale=1):
                                        gr.Text("DC(Dead Cell)", show_label=False)
                                    with gr.Row(scale=2):
                                        analysis_protocol_DC = gr.Radio(
                                        choices=["ON", "OFF"], label=None, show_label=False
                                    )
            
            
        with gr.Row(): # 버튼 
            with gr.Row():
                analysis_start_button = gr.Button("START", elem_id="analysis-button")
                analysis_cancel_button = gr.Button("CANCEL", elem_id="analysis-button")
            with gr.Row(visible=False) as go_second_section:
                with gr.Row():
                    go_analysis_data_page2 = gr.Button("Analysis Data")
        
        ####################################################### 팝업 창 오버레이 #######################################################################
        page1_overlay = gr.Markdown("", elem_id="overlay", visible=False)
        start_popup = gr.Column(visible=False, elem_id="popup-container")
        cancel_popup = gr.Column(visible=False, elem_id="popup-container")
        with start_popup:
            gr.Markdown("### 분석을 진행할까요? \n ### 시간이 오래 걸릴 수 있습니다.")
            with gr.Row():
                start_yes_button = gr.Button("예")
                start_no_button = gr.Button("아니오")
        with cancel_popup:
            gr.Markdown("### 정말로 중단하시나요? \n ### 프로그램이 중단됩니다.")
            with gr.Row():
                cancel_yes_button = gr.Button("예")
                cancel_no_button = gr.Button("아니오")
    
    
    
######################### 
#######  page 2 #########
#########################
    original_gallery_image = gr.State() # selected_well 선택 시 오버레이를 위해 원본 갤러리 이미지 저장
    overlay_gallery_image = gr.State()
    
    with gr.Column(visible=False) as page2:
                # Gradio 상태 변수
        stack_index = gr.State(0)
        export_select_well_toggle_state = gr.State(False)
        selected_idx_for_export = gr.State([])
        
        with gr.Column():
            with gr.Row(): # 토글 버튼 및 팝업 창 
                show_DIC = Toggle(label="DIC", value=True, interactive=True, elem_classes="toggle-wrapper")
                show_graph = Toggle(label="Graph", value=False, interactive=True, elem_classes="toggle-wrapper")
                select_well_button = Toggle(label="Select Well", elem_id="toggle-wrapper")
                
                with gr.Column():
                    page2_export_button = gr.Button("Export Data", elem_id="page2-button")
                    page2_export_popup = gr.Column(visible=False, elem_id="export-popup")
                    page2_overlay = gr.Markdown("", elem_id="overlay", visible=False)
                    with page2_export_popup:
                        with gr.Row():
                            with gr.Column():
                                page2_export_what_well = gr.Radio(["All Wells", "Selected Wells"], value="All Wells", show_label=False, interactive=False)

                        gr.Markdown("---")

                        with gr.Column():
                            page2_vis_main = gr.Radio(["Visualization Images"], show_label=False, interactive=True)
                            page2_vis_format = gr.Radio(["TIFF", "PNG", "JPEG"], show_label=False, interactive=True)

                        with gr.Column():
                            page2_graph_main = gr.Radio(["Graph"], show_label=False, interactive=True)
                            page2_graph_format = gr.Radio(["TIFF", "PNG", "JPEG"], show_label=False, interactive=True)

                        page2_save_button = gr.Button("Save")

        ###################################### 갤러리 컴포넌트 구성 ###################################################
        with gr.Column(visible=False) as well_06: # 06 well gallery
            cols = ["A","B"]  # A,B

            with gr.Row():
                with gr.Row(scale=0):
                    pass
                with gr.Row(scale=1):
                    for row in range(1, 4):
                        gr.HTML(f"""<div style='font-size: 30px; font-weight: bold; text-align: left; margin: 0 auto; width: 100px'> {row} </div>""")
            # 실제 갤러리 (3열, 2행)
            with gr.Row():
                with gr.Column(scale=0.05, min_width=20):

                    for col in cols:
                        gr.HTML(f"<div style='margin-top: 280px; font-size: 30px; font-weight: bold;'> {col} </div>")
                        gr.HTML(f"<div style='margin-top: 150px; font-size: 30px; font-weight: bold;'></div>")
                with gr.Column(scale=0.95):
                    well_gallery_06 = gr.Gallery(
                            label="06-Well Plate",
                            show_label=False,
                            columns=3,
                            height=1200,
                            allow_preview=False,
                        )

        with gr.Column(visible=False) as well_12: # 12 well gallery
            cols = ["A","B", "C"]  # A,B,C

            with gr.Row():
                with gr.Row(scale=0):
                    pass
                with gr.Row(scale=1):
                    for row in range(1, 5):
                        gr.HTML(f"""<div style='font-size: 30px; font-weight: bold; text-align: left; margin: 0 auto; width: 100px'> {row} </div>""")
            # 실제 갤러리 (3열, 2행)
            with gr.Row():
                with gr.Column(scale=0.05, min_width=20):

                    for col in cols:
                        gr.HTML(f"<div style='margin-top: 200px; font-size: 30px; font-weight: bold;'> {col} </div>")
                        gr.HTML(f"<div style='margin-top: 100px; font-size: 30px; font-weight: bold;'></div>")

                with gr.Column(scale=0.95):
                    well_gallery_12 = gr.Gallery(
                            label="12-Well Plate",
                            show_label=False,
                            columns=4,
                            height=1200,
                            allow_preview=False,
                        )

        with gr.Column(visible=False) as well_24: # 24 well gallery
            cols = ["A","B", "C", "D"]  # A,B,C,D

            with gr.Row():
                with gr.Row(scale=0):
                    pass
                with gr.Row(scale=1):
                    for row in range(1, 7):
                        gr.HTML(f"""<div style='font-size: 30px; font-weight: bold; text-align: left; margin: 0 auto; width: 100px'> {row} </div>""")
                        
            # 실제 갤러리 (3열, 2행)
            with gr.Row():
                with gr.Column(scale=0.05, min_width=20):

                    for col in cols:
                        gr.HTML(f"<div style='margin-top: 140px; font-size: 30px; font-weight: bold;'> {col} </div>")
                        gr.HTML(f"<div style='margin-top: 30px; font-size: 30px; font-weight: bold;'></div>")

                with gr.Column(scale=0.95):
                    well_gallery_24 = gr.Gallery(
                            label="24-Well Plate",
                            show_label=False,
                            columns=6,
                            height=1200,
                            allow_preview=False,
                        )

        with gr.Column(visible=False) as well_96: # 96 well gallery
            cols = [chr(i) for i in range(ord("A"), ord("I"))]  # A~H

            with gr.Row():
                with gr.Row(scale=0):
                    pass
                with gr.Row(scale=1):
                    for row in range(1, 13):
                        gr.HTML(f"""<div style='font-size: 30px; font-weight: bold; text-align: left; margin: 0 auto; width: 100px'> {row} </div>""")
            # 실제 갤러리 (12열, 8행)
            with gr.Row():
                with gr.Column(scale=0.05, min_width=20):

                    for col in cols:
                        gr.HTML(f"<div style='margin-top: 35px; font-size: 30px; font-weight: bold;'> {col} </div>")
                        gr.HTML(f"<div style='margin-top: 0px; font-size: 30px; font-weight: bold;'></div>")

                with gr.Column(scale=0.95):
                    well_gallery_96 = gr.Gallery(
                            label="96-Well Plate",
                            show_label=False,
                            columns=12,
                            height=1200,
                            allow_preview=False,
                        )
            
        ###################################### 1페이지로 전환 ################################################
        with gr.Row():
            # 1페이지로 변경(최대/최소 10px)
            with gr.Column(elem_id="col2"):
                second_to_first_page = gr.Button("HOME") 
                second_to_first_page.click(fn=switch_page, outputs=[page1, page2])
            # 왼쪽 이미지 - 첫 번째 컬럼(최대/최소 10px)
            with gr.Column(elem_id="col1"):
                page2_left_button = gr.Button("LEFT")
                
            # 오른쪽 이미지 - 세 번째 컬럼(최대/최소 10px)  
            with gr.Column(elem_id="col3"):
                page2_right_button = gr.Button("RIGHT")
    
            # 네 번째 컬럼(남은 공간)
            with gr.Column(elem_id="col4"):
                with gr.Row():
                    #gr.Markdown("## A")
                    page2_time_slider = gr.Slider(minimum=1, maximum=24, value=1, step=1, interactive=False, show_reset_button=False)
        


######################### 
#######  page 3 #########
#########################
    with gr.Column(visible=False) as page3:
        selected_well_folder = gr.State("")  # 선택된 웰 폴더 경로
        selected_well_label = gr.State("")
        
        with gr.Column(): # export 버튼 및 팝업 창 
            with gr.Row():
                with gr.Column(scale=1, min_width=0):
                    pass
                with gr.Column(scale=0, min_width=150):
                    page3_export_button = gr.Button("Export Data")

                    page3_export_popup = gr.Column(visible=False, elem_id="export-popup")
                    page3_overlay = gr.Markdown("", elem_id="overlay", visible=False)
                    with page3_export_popup:
                        with gr.Row():
                            gr.Markdown("## Current Well")
                        gr.Markdown("---")

                        with gr.Column():
                            page3_vis_main = gr.Radio(["Visualization Images"], show_label=False, interactive=True)
                            page3_vis_format = gr.Radio(["TIFF", "PNG", "JPEG"], show_label=False, interactive=True)

                        with gr.Column():
                            page3_graph_main = gr.Radio(["Graph"], show_label=False, interactive=True)
                            page3_graph_format = gr.Radio(["TIFF", "PNG", "JPEG"], show_label=False, interactive=True)

                        page3_save_button = gr.Button("Save")

        with gr.Row(): 
            with gr.Column(scale=0): 
                selected_well_markdown = gr.HTML(f"""<div style='font-size: 80px; font-weight: bold; text-align: left;'> {selected_well_label} </div>""")

            with gr.Column(): 
                with gr.Row():  # Pseudo Color 토글
                    with gr.Column(scale=1):
                        pass
                    with gr.Column(scale=1):
                        pseudo_color_toggle = Toggle(label="Pseudo Color", value=False, interactive=True)

                with gr.Row():  # SC, DC 토글
                    SC_toggle = Toggle(label="SC", value=False, interactive=True)
                    DC_toggle = Toggle(label="DC", value=False, interactive=True)


            ######################################### peak time frame ########################################################3
            df = pd.DataFrame(peak_data)
            df_component = gr.DataFrame(df, headers=["", "Peak Count", "Peak Time"], datatype=["str", "str", "str"])

        ########################################### DIC, Graph 시각화 ######################################################
        with gr.Row():
            DIC_image = gr.Image(show_label=False, sources=[])
            ICD_graph = gr.Plot(show_label=False)
    
    ############################################ 하단 컨트롤 버튼 ###########################################################
        with gr.Row():
            # 2페이지로 변경 
            with gr.Column(elem_id="col0"):
                back_second_page = gr.Button("<")
                back_second_page.click(fn=switch_page, outputs=[page2, page3])
            # 1페이지로 변경(최대/최소 10px)
            with gr.Column(elem_id="col2"):
                third_to_first_page = gr.Button("HOME") 
                third_to_first_page.click(fn=switch_page, outputs=[page1, page3])
            # 왼쪽 이미지 - 첫 번째 컬럼(최대/최소 10px)
            with gr.Column(elem_id="col1"):
                page3_left_button = gr.Button("LEFT")
                
            # 오른쪽 이미지 - 세 번째 컬럼(최대/최소 10px)  
            with gr.Column(elem_id="col3"):
                page3_right_button = gr.Button("RIGHT")
    
            # 네 번째 컬럼(남은 공간)
            with gr.Column(elem_id="col4"):
                with gr.Row():
                    #gr.Markdown("🢀 🢂 | 0000# (시간)")
                    page3_time_slider = gr.Slider(minimum=1, maximum=24, value=1, step=1, interactive=False, show_reset_button=False)
           

           
############################
# page 1 콜백 연결
############################

    information_plate_type.change(fn=plate_gallery_visible, inputs=[information_plate_type], outputs=[well_06, well_12, well_24, well_96])
    # (1) Total/Interval 변경 시 Cycle 자동 계산
    information_total_time_hour.change(cycle_auto_calc, inputs=[information_total_time_hour, information_total_time_min, information_interval_hour, information_interval_min], outputs=cycle_auto_calc_text)
    information_total_time_min.change(cycle_auto_calc,  inputs=[information_total_time_hour, information_total_time_min, information_interval_hour, information_interval_min], outputs=cycle_auto_calc_text)
    information_interval_hour.change(cycle_auto_calc,  inputs=[information_total_time_hour, information_total_time_min, information_interval_hour, information_interval_min], outputs=cycle_auto_calc_text)
    information_interval_min.change(cycle_auto_calc,  inputs=[information_total_time_hour, information_total_time_min, information_interval_hour, information_interval_min], outputs=cycle_auto_calc_text)
    cycle_auto_calc_text.change(fn=update_slider, inputs=[cycle_auto_calc_text], outputs=[page2_time_slider])
    cycle_auto_calc_text.change(fn=update_slider, inputs=[cycle_auto_calc_text], outputs=[page3_time_slider])
    
    # (2) START 버튼 클릭 시 팝업 표시 및 progress_bar 실행
    analysis_start_button.click(fn=show_popup, outputs=[start_popup, page1_overlay])
    start_yes_button.click(hide_popup, outputs=[start_popup, page1_overlay]).then(
                            fn=create_2d_mapping, inputs=[selected_folder, import_selected_folder], outputs=[well_stack_mapping_path]).then(
                                fn=progress_bar,inputs=[well_stack_mapping_path, import_selected_folder], outputs=[model_processing_time, go_second_section]).then(fn=all_image_open, inputs=[well_stack_mapping_path], outputs=[well_stack_mapping_image]) # 개선사항 : all_image_open도 progress bar에 반영해야함
                                # create_2d_mapping에서 import folder면 import_selected_folder 내의 이미지를 가져오도록 설정
                                # progress_bar에서 inference를 하는데 만약 import folder이면 inference 안함
    start_no_button.click(hide_popup, outputs=[start_popup, page1_overlay])

    # (3) CANCEL 버튼 클릭 시 팝업 표시
    analysis_cancel_button.click(fn=show_popup, outputs=[cancel_popup, page1_overlay])
    cancel_yes_button.click(hide_popup, outputs=[cancel_popup, page1_overlay])
    cancel_no_button.click(hide_popup, outputs=[cancel_popup, page1_overlay])

    # (4) Analysis Data 버튼 클릭 시 2페이지로 이동
    #     그리고 2D 매핑 생성 + 갤러리 업데이트
    go_analysis_data_page2.click(fn=switch_page, outputs=[page2, page1])
    
    
############################
# page 2 콜백 연결
############################
    
    # 1페이지에서 분석 버튼 누르면 바로 초점 맞는 이미지 저장 + 갤러리 업데이트
    go_analysis_data_page2.click(fn=update_well_gallery, inputs=[well_stack_mapping_image, stack_index], outputs=[well_gallery_06, well_gallery_12, well_gallery_24, well_gallery_96, original_gallery_image, overlay_gallery_image]) # 갤러리에 업데이트 되면 update_well_gallery 함수에서 바로 폴더 저장하는 함수 불러옴
    
    # DIC, Graph 간의 토글 버튼 상호 작용 -> 토글이 되면 해당 이미지로 전환해야한다!
    show_DIC.change(fn=show_DIC_or_Graph, outputs=[show_graph, show_DIC])
    show_graph.change(fn=show_DIC_or_Graph, outputs=[show_DIC, show_graph])
    
    # selected well 누르면 export 내의 버튼이 바뀜
    select_well_button.change(fn=is_selected_toggle, inputs=[export_select_well_toggle_state, selected_idx_for_export], outputs=[export_select_well_toggle_state, page2_export_what_well, selected_idx_for_export]).then(fn=overlay_image_init, inputs=[original_gallery_image], outputs=[overlay_gallery_image, well_gallery_06, well_gallery_12, well_gallery_24, well_gallery_96]) # select well 토글이 되면 오버레이 이미지 갤러리 변수가 원본 이미지 갤러리로 바뀐다
    
    # 좌우 클릭 시 interval이 넘어가고 그에 따른 갤러리 이미지도 바뀐다.
    page2_left_button.click(fn=prev_stack, inputs=stack_index, outputs=stack_index).then(fn=slider_position_change, inputs=[stack_index], outputs=[page2_time_slider]).then(fn=slider_position_change, inputs=[stack_index], outputs=[page3_time_slider])
    page2_right_button.click(fn=next_stack, inputs=[stack_index, cycle_auto_calc_text], outputs=stack_index).then(fn=slider_position_change, inputs=[stack_index], outputs=[page2_time_slider]).then(fn=slider_position_change, inputs=[stack_index], outputs=[page3_time_slider])
    stack_index.change(fn=update_well_gallery, inputs=[well_stack_mapping_image, stack_index], outputs=[well_gallery_06, well_gallery_12, well_gallery_24, well_gallery_96, original_gallery_image, overlay_gallery_image])
    
    # export well 내 디렉토리에 저장
    page2_save_button.click(fn=hide_popup, outputs=[page2_export_popup, page2_overlay]).then(fn=page2_export_data_save, inputs=[well_stack_mapping_path, stack_index, page2_export_what_well, page2_vis_main, page2_vis_format, page2_graph_main, page2_graph_format], outputs=[page2_export_what_well, page2_vis_main, page2_vis_format, page2_graph_main, page2_graph_format])
    
    # 갤러리에서 아이템 선택 시 3페이지 이동 -> well 마다 각각 구현함 // but, selected_well toggle이 눌리면 3페이지로 이동 안하고 클릭 이벤트만 저장할 것 -> 추후 export button이 눌리면 그때 클릭 이벤트를 넘겨줄 예정
    well_gallery_06.select(fn=on_gallery_select, inputs=[well_stack_mapping_path, original_gallery_image, overlay_gallery_image, stack_index, export_select_well_toggle_state, selected_idx_for_export, pseudo_color_toggle], outputs=[selected_well_folder, selected_well_label, stack_index, DIC_image, selected_well_markdown, selected_idx_for_export, well_gallery_06, page2_export_what_well]).then(
        fn=update_Graph_and_Peak, inputs=[selected_well_label, cycle_auto_calc_text, stack_index], outputs=[ICD_graph, df_component]).then(fn=switch_page, inputs=[export_select_well_toggle_state], outputs=[page3, page2])
    
    well_gallery_12.select(fn=on_gallery_select, inputs=[well_stack_mapping_path, original_gallery_image, overlay_gallery_image, stack_index, export_select_well_toggle_state, selected_idx_for_export, pseudo_color_toggle], outputs=[selected_well_folder, selected_well_label, stack_index, DIC_image, selected_well_markdown, selected_idx_for_export, well_gallery_12, page2_export_what_well]).then(
        fn=update_Graph_and_Peak, inputs=[selected_well_label, cycle_auto_calc_text, stack_index], outputs=[ICD_graph, df_component]).then(fn=switch_page, inputs=[export_select_well_toggle_state], outputs=[page3, page2])
    
    well_gallery_24.select(fn=on_gallery_select, inputs=[well_stack_mapping_path, original_gallery_image, overlay_gallery_image, stack_index, export_select_well_toggle_state, selected_idx_for_export, pseudo_color_toggle], outputs=[selected_well_folder, selected_well_label, stack_index, DIC_image, selected_well_markdown, selected_idx_for_export, well_gallery_24, page2_export_what_well]).then(
        fn=update_Graph_and_Peak, inputs=[selected_well_label, cycle_auto_calc_text, stack_index], outputs=[ICD_graph, df_component]).then(fn=switch_page, inputs=[export_select_well_toggle_state], outputs=[page3, page2])
    
    well_gallery_96.select(fn=on_gallery_select, inputs=[well_stack_mapping_path, original_gallery_image, overlay_gallery_image, stack_index, export_select_well_toggle_state, selected_idx_for_export, pseudo_color_toggle], outputs=[selected_well_folder, selected_well_label, stack_index, DIC_image, selected_well_markdown, selected_idx_for_export, well_gallery_96, page2_export_what_well]).then(
        fn=update_Graph_and_Peak, inputs=[selected_well_label, cycle_auto_calc_text, stack_index], outputs=[ICD_graph, df_component]).then(fn=switch_page, inputs=[export_select_well_toggle_state], outputs=[page3, page2])
   

############################
# page 3 콜백 연결
############################

    # 팝업 표시/숨김 버튼 콜백
    page3_export_button.click(fn=show_popup, outputs=[page3_export_popup, page3_overlay]); page2_export_button.click(fn=show_popup, outputs=[page2_export_popup, page2_overlay])
    page3_save_button.click(fn=hide_popup, outputs=[page3_export_popup, page3_overlay]).then(fn=page3_export_data_save, inputs=[DIC_image, ICD_graph, stack_index, selected_well_label, selected_well_folder, page3_vis_main, page3_vis_format, page3_graph_main, page3_graph_format], outputs=[page3_vis_main, page3_vis_format, page3_graph_main, page3_graph_format])


    # # - Back 버튼 -> third_page_stack_index - 1 후 DIC 이미지 업데이트 + 그래프에서 점선 위치 변경
    page3_left_button.click(fn=prev_stack, inputs=stack_index, outputs=stack_index).then(fn=slider_position_change, inputs=[stack_index], outputs=[page3_time_slider]).then(fn=slider_position_change, inputs=[stack_index], outputs=[page2_time_slider]).then(
        fn=update_DIC_image, inputs=[pseudo_color_toggle, selected_well_folder, stack_index], outputs=DIC_image).then(fn=update_Graph_and_Peak, inputs=[selected_well_label, cycle_auto_calc_text, stack_index], outputs=[ICD_graph, df_component])

    # # - Next 버튼 -> third_page_stack_index + 1 후 DIC 이미지 업데이트 + 그래프에서 점선 위치 변경
    page3_right_button.click(fn=next_stack, inputs=[stack_index, cycle_auto_calc_text], outputs=stack_index).then(fn=slider_position_change, inputs=[stack_index], outputs=[page3_time_slider]).then(fn=slider_position_change, inputs=[stack_index], outputs=[page2_time_slider]).then(
        fn=update_DIC_image, inputs=[pseudo_color_toggle, selected_well_folder, stack_index], outputs=DIC_image).then(fn=update_Graph_and_Peak, inputs=[selected_well_label, cycle_auto_calc_text, stack_index], outputs=[ICD_graph, df_component])
    
    # psudo color가 토글 되면 GFP + RFP 병합된 이미지가 보여야한다.
    pseudo_color_toggle.change(fn=merge_GFP_RFP, inputs=[pseudo_color_toggle, selected_well_folder, stack_index], outputs=[DIC_image]) # 현재 위치의 정보를 input으로 / output : 병합 이미지