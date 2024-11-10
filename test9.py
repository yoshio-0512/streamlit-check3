import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import torch
import cv2

# モデルのロード
MODEL_PATH = "e_meter_segadd2.pt"  # モデルファイルのパス
device = torch.device("cpu")  # 必要に応じて "cuda" に変更

# YOLOv5モデルをロード
@st.cache_resource
def load_model(model_path, device):
    model_data = torch.load(model_path, map_location=device)
    model = model_data['model'].float().fuse().eval()  # モデルを評価モードに
    return model

model = load_model(MODEL_PATH, device)

# ヘルパー関数
def preprocess_image(image):
    """画像をリサイズしてYOLOの入力形式に変換"""
    square_image = expand_image_to_square(image)
    resized_image = square_image.resize((640, 640))
    return np.array(resized_image)

def expand_image_to_square(image, background_color=(255, 255, 255)):
    """画像を正方形に変換し、背景を塗りつぶす"""
    width, height = image.size
    if width == height:
        return image
    new_size = max(width, height)
    result = Image.new(image.mode, (new_size, new_size), background_color)
    result.paste(image, ((new_size - width) // 2, (new_size - height) // 2))
    return result

def draw_boxes(image, detections):
    """検出結果を画像に描画"""
    draw = ImageDraw.Draw(image)
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{cls} {conf:.2f}", fill="red")
    return image

# Streamlitアプリケーション
st.title("YOLOv5での物体検出")
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("またはカメラで画像を撮影してください")

if uploaded_file or camera_image:
    # 入力画像をPillow形式で読み込む
    input_image = Image.open(uploaded_file or camera_image)
    st.image(input_image, caption="入力画像", width=300)

    if st.button("物体検出を開始"):
        with st.spinner("物体を検出中..."):
            try:
                # 画像を前処理
                input_image_cv2 = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

                # 推論を実行
                results = model(input_image_cv2)

                # 検出結果の取得
                detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]

                if len(detections) == 0:
                    st.error("物体が検出されませんでした。", icon="🚨")
                else:
                    # 検出結果を描画
                    result_image = draw_boxes(input_image.copy(), detections)
                    st.image(result_image, caption="検出結果", width=300)
                    st.download_button("結果画像をダウンロード", data=result_image.tobytes(), file_name="result.png", mime="image/png")

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}", icon="❌")
