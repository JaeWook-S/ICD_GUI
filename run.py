import os
import gradio.utils as gr_utils

# 패치 적용 전에 safe_join 함수 확인 (디버깅용)
#print("Before patch:", gr_utils.safe_join)

def patched_safe_join(root, *paths):
    # 만약 첫 번째 경로가 절대 경로라면, root 기준 상대 경로로 변환
    if paths and os.path.isabs(paths[0]):
        rel_path = os.path.relpath(paths[0], os.path.abspath(root))
        paths = (rel_path,) + paths[1:]
    # 단순히 os.path.join을 호출하여 합침 (재귀 호출 제거)
    return os.path.join(root, *paths)

# safe_join 함수를 완전히 대체합니다.
gr_utils.safe_join = patched_safe_join

# 패치 적용 후 safe_join 함수 확인 (디버깅용)
#print("After patch:", gr_utils.safe_join)

# 이후에 모듈 재로딩 등 진행
import importlib
import gradio.components.file_explorer as fe
importlib.reload(fe)

# 그 후 나머지 모듈들 임포트
import gradio as gr
import webbrowser

from all_page import main_page

css = """
/* first page css */

#information-content {
    display: flex; /* 가로 정렬 */
    align-items: center; /* 세로 중앙 정렬 */
    gap: 0px !important; /* 요소 간 간격 제거 */
    margin: 0px !important; /* 위아래 간격 제거 */
    padding: 0px 0 !important; /* 위아래 패딩 최소화 */
    border: 0.5px solid black !important;  /* 검은색 테두리 */
    border-radius: 5px;  /* 둥근 모서리 (원하면 제거 가능) */
}

#upload-content {
    height: 320px !important;
    background-color: black !important;  /* 배경 파란색 */
    color: white !important; /* 글씨 흰색 */
    border: 1px solid black !important;  /* 검은색 테두리 */
    border-radius: 5px;  /* 둥근 모서리 (원하면 제거 가능) */
}

.progress-text {
    background-color: black !important; /* 배경 검은색 */
    color: white !important; /* 글씨 흰색 */
    height: 125px !important;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    border: 1px solid black;
    border-radius: 5px;
    /* 세로 중앙 정렬을 위한 flexbox 사용 */
    display: flex;
    align-items: center;  /* 세로 중앙 정렬 */
    justify-content: center;  /* 가로 중앙 정렬 */
}


#analysis-button {
    width: 150px !important;    
    min-width: 150px !important;
    max-width: 150px !important;
    /*background-color: yellow !important;   배경 노란색 */
    color: black !important;  /* 글자색 검정 */
    border: 2px solid black !important;  /* 검은색 테두리 */
    font-weight: bold !important;  /* 글자를 굵게 */
}

#right-align {
    display: flex;
    justify-content: flex-end; /* 우측 정렬 */
    min-width: 150px !important;
    max-width: 150px !important;
    gap: 10px; /* 요소 사이 간격 */
}

#center-align {
    text-align: left;
}

#popup-container {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
    width: 300px;
    text-align: center;
    z-index: 1000;
}

#overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: 999;
}

/* second page css */


#page2-button {
    color: black !important;  /* 글자색 검정 */
    border: 2px solid black !important;  /* 검은색 테두리 */
    font-weight: bold !important;  /* 글자를 굵게 */
}


/* third page css */

#export-popup {
    position: fixed;
    top: 10%;   /* 상단에서 5% 위치 */
    right: 5%; /* 오른쪽에서 5% 위치 */
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
    width: 300px;
    text-align: center;
    z-index: 1000;
}

/* 1~3번 컬럼( col1, col2, col3 )은 10px 폭으로 강제 */
#col0, #col1, #col2, #col3 {
    max-width: 50px !important;
    min-width: 50px !important;
    width: 50px !important;
    flex: 0 0 auto !important; /* flex 확장 방지 */
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}
/* Textbox 내부의 라벨, 마진, 패딩 등 제거 */
#col0 .label, #col1 .label, #col2 .label, #col3 .label {
    display: none !important; /* 라벨 숨김 */
}
#col0 .wrap, #col1 .wrap, #col2 .wrap, #col3 .wrap {
    padding: 0 !important;
    margin: 0 !important;
}
/* 마지막 컬럼( col4 )은 flex:1로 남은 공간을 전부 사용 */
#col4 {
    flex: 1 !important;
}


/* 1, 2, 3번 컬럼에 들어있는 버튼들의 (hover 포함) 배경, 테두리, 그림자 제거 */
#col0 .gr-button, #col1 .gr-button, #col2 .gr-button, #col3 .gr-button {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    border-radius: 0 !important; /* 모서리 둥글기도 제거 */
    margin: 0 !important;
    padding: 0 !important;
}
#col0 .gr-button:hover, #col1 .gr-button:hover, #col2 .gr-button:hover, #col3 .gr-button:hover {
    background-color: transparent !important;
}

#my_slider input[type="range"]::-webkit-slider-runnable-track {
    height: 8px;
    border-radius: 8px;
    background: repeating-linear-gradient(
        to right,
        black 0px,
        black 5px,
        transparent 5px,
        transparent 10px
    );
}
#my_slider input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #ccc;
    margin-top: -4px;
}

"""



# 강제 라이트 모드 URL
url = "http://127.0.0.1:7860/?__theme=light"

def main():
    with gr.Blocks(css=css) as demo:

        main_page()     

    webbrowser.open(url)
    demo.launch(share=True, debug=True)

if __name__ == "__main__":
    main()