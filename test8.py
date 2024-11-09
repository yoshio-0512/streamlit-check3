import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# --------------------------------------------------------------
# ヘルパー関数
# --------------------------------------------------------------

def expand_image_to_square(image, background_color=(255, 255, 255)):
    """画像を正方形に変換し、背景を塗りつぶす"""
    width, height = image.size
    if width == height:
        return image
    new_size = max(width, height)
    result = Image.new(image.mode, (new_size, new_size), background_color)
    result.paste(image, ((new_size - width) // 2, (new_size - height) // 2))
    return result

def preprocess_image(image):
    """画像の前処理を行う"""
    square_image = expand_image_to_square(image)
    resized_image = square_image.resize((416, 416))
    return resized_image

def find_top_bottom(mask_image):
    """画像マスクから上端と下端の座標を検出"""
    binary_image = (mask_image > 128).astype(np.uint8) * 255
    rows, cols = binary_image.shape
    # 上端の検出
    for i in range(rows):
        for j in range(cols - 1, -1, -1):
            if binary_image[i, j] == 255:
                top = (i, j)
                break
        if 'top' in locals():
            break
    # 下端の検出
    for i in range(rows - 1, -1, -1):
        for j in range(cols):
            if binary_image[i, j] == 255:
                bottom = (i, j)
                break
        if 'bottom' in locals():
            break
    return top, bottom

def classify_wiring(sorted_top, sorted_bottom):
    """配線の分類を行う"""
    center_x = sum([coords[1] for coords in sorted_bottom]) / 4
    if all(sorted_bottom[::2, 1] < center_x) and all(sorted_bottom[1::2, 1] > center_x):
        return "左電源の正結線の可能性が高いです", "✅"
    elif all(sorted_bottom[::2, 1] > center_x) and all(sorted_bottom[1::2, 1] < center_x):
        return "右電源の正結線の可能性が高いです", "✅"
    return "誤配線の可能性があります。目視で確認してください", "⚠"

def draw_detected_points(image, sorted_top, sorted_bottom):
    """検出された点を画像上に描画"""
    draw = ImageDraw.Draw(image)
    for i, (y, x) in enumerate(sorted_top):
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 0, 255))
    for i, (y, x) in enumerate(sorted_bottom):
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(0, 0, 255))
    return image

# --------------------------------------------------------------
# Streamlit アプリケーション
# --------------------------------------------------------------

st.title("画像の物体検出と配線チェック")
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("またはカメラで画像を撮影してください")

if uploaded_file or camera_image:
    input_image = Image.open(uploaded_file or camera_image)
    processed_image = preprocess_image(input_image)
    st.image(processed_image, caption="前処理済み画像", width=300)

    if st.button("物体検出を開始"):
        with st.spinner("物体を検出中..."):
            model = YOLO("e_meter_segadd2.torchscript")
            results = model.predict(processed_image, imgsz=416, conf=0.5, classes=0)

            if not results[0].masks:
                st.error("配線の検出に失敗しました。目視で確認してください", icon="🚨")
            else:
                mask_images = [mask.data[0].cpu().numpy() * 255 for mask in results[0].masks]
                coordinates = [find_top_bottom(mask) for mask in mask_images]
                top_list = np.array([coord[0] for coord in coordinates])
                bottom_list = np.array([coord[1] for coord in coordinates])

                # ソート
                sorted_top = np.array(sorted(top_list, key=lambda x: x[1]))
                sorted_bottom = np.array([bottom for _, bottom in sorted(zip(top_list, bottom_list), key=lambda x: x[0][1])])

                # 分類
                message, icon = classify_wiring(sorted_top, sorted_bottom)
                st.success(message, icon=icon)

                # 描画
                result_image = draw_detected_points(processed_image.copy(), sorted_top, sorted_bottom)
                st.image(result_image, caption="検出結果", width=300)
                st.download_button("結果画像をダウンロード", data=result_image.tobytes(), file_name="result.png", mime="image/png")
