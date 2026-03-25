import streamlit as st
import numpy as np
from PIL import Image
from src.face_processor import FaceProcessor

st.set_page_config(page_title="人脸识别", layout="wide")
st.title("🔍 人脸识别系统")

# 初始化处理器（自动加载known_faces/）
processor = FaceProcessor("known_faces")

# 侧边栏
with st.sidebar:
    st.header("配置")
    tolerance = st.slider("识别阈值", 0.3, 0.8, 0.6, 0.05)
    st.write(f"人脸库: {len(processor.known_names)}人")
    for name in processor.known_names:
        st.write(f"• {name}")

# 主界面
uploaded = st.file_uploader("上传图片", type=['jpg','jpeg','png','bmp'])
if uploaded:
    image = Image.open(uploaded)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原图")
        st.image(image, use_container_width=True)
    
    if st.button("开始处理", type="primary"):
        with st.spinner("处理中..."):
            np_img = np.array(image.convert('RGB'))
            results = processor.process(np_img, tolerance=tolerance)
            result_img = processor.draw_results(image.copy(), results)
            
            with col2:
                st.subheader(f"结果（{results['count']}个人脸）")
                st.image(result_img, use_container_width=True)
            
            # 显示详细信息
            for i, (name, conf) in enumerate(zip(results['names'], results['confidences'])):
                name_str = name or "未知"
                conf_str = f"{conf:.1%}" if conf else "N/A"
                st.write(f"人脸{i+1}: **{name_str}** (置信度: {conf_str})")
