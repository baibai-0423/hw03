"""
人脸识别系统 - 单文件完整版
包含：Streamlit界面 + 人脸检测/识别逻辑 + 128维特征提取
"""

import streamlit as st
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import os
from typing import List, Tuple, Optional, Dict


# ============ 人脸处理类（原src/的内容） ============

class FaceProcessor:
    """人脸处理器：检测 + 128维特征编码 + 识别"""
    
    def __init__(self, known_faces_dir: str = None):
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        
        if known_faces_dir and os.path.exists(known_faces_dir):
            self._load_known_faces(known_faces_dir)
    
    def _load_known_faces(self, directory: str) -> None:
        """加载已知人脸库"""
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(directory, filename)
                try:
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        name = os.path.splitext(filename)[0]
                        self.known_names.append(name)
                except Exception as e:
                    st.warning(f"加载失败 {filename}: {e}")
    
    def detect_faces(self, image: np.ndarray) -> Tuple[List[Tuple], List[np.ndarray]]:
        """检测人脸位置并提取128维特征"""
        face_locations = face_recognition.face_locations(image, model="hog")
        face_encodings = face_recognition.face_encodings(image, face_locations)
        return face_locations, face_encodings
    
    def recognize_faces(self, face_encodings: List[np.ndarray], tolerance: float = 0.6) -> List[Optional[str]]:
        """与已知人脸库比对识别"""
        if not self.known_encodings:
            return [None] * len(face_encodings)
        
        results = []
        for encoding in face_encodings:
            distances = face_recognition.face_distance(self.known_encodings, encoding)
            best_idx = np.argmin(distances)
            if distances[best_idx] <= tolerance:
                results.append(self.known_names[best_idx])
            else:
                results.append(None)
        return results
    
    def process(self, image: np.ndarray, recognize: bool = True, tolerance: float = 0.6) -> Dict:
        """完整处理流程：检测+识别"""
        locations, encodings = self.detect_faces(image)
        names = [None] * len(locations)
        confidences = [None] * len(locations)
        
        if recognize and self.known_encodings and encodings:
            names = self.recognize_faces(encodings, tolerance)
            for i, enc in enumerate(encodings):
                dists = face_recognition.face_distance(self.known_encodings, enc)
                confidences[i] = round(1 - np.min(dists), 3) if len(dists) > 0 else 0
        
        return {
            'count': len(locations),
            'locations': locations,
            'names': names,
            'confidences': confidences,
            'encodings': encodings
        }


# ============ 工具函数 ============

def draw_results(image: Image.Image, results: Dict, show_confidence: bool = True) -> Image.Image:
    """在图片上绘制检测结果"""
    draw = ImageDraw.Draw(image)
    for i, (top, right, bottom, left) in enumerate(results['locations']):
        # 画框
        draw.rectangle([left, top, right, bottom], outline="#00FF00", width=3)
        
        # 标签
        name = results['names'][i] or "未知"
        label = f"{name} ({results['confidences'][i]:.1%})" if show_confidence and results['confidences'][i] else name
        
        # 文字背景
        bbox = draw.textbbox((0, 0), label)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([left, top-text_h-10, left+text_w+10, top], fill="#00FF00")
        draw.text((left+5, top-text_h-5), label, fill="black")
    
    return image


# ============ Streamlit界面 ============

def main():
    st.set_page_config(page_title="人脸识别", layout="wide")
    st.title("🔍 人脸识别系统")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("配置")
        mode = st.radio("模式", ["仅检测", "检测+识别"])
        tolerance = st.slider("识别阈值", 0.3, 0.8, 0.6, 0.05) if mode == "检测+识别" else 0.6
        
        # 人脸库信息
        processor = FaceProcessor("known_faces" if os.path.exists("known_faces") else None)
        if processor.known_names:
            st.success(f"人脸库: {len(processor.known_names)}人")
            for n in processor.known_names:
                st.write(f"• {n}")
        elif mode == "检测+识别":
            st.info("提示：创建 known_faces/ 文件夹并放入人脸图片可启用识别")
    
    # 主界面
    uploaded = st.file_uploader("上传图片", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded:
        image = Image.open(uploaded)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("原始图片")
            st.image(image, use_container_width=True)
        
        if st.button("开始处理", type="primary"):
            with st.spinner("处理中..."):
                # 处理
                np_img = np.array(image.convert('RGB'))
                do_recognize = (mode == "检测+识别")
                results = processor.process(np_img, recognize=do_recognize, tolerance=tolerance)
                
                # 显示结果
                result_img = draw_results(image.copy(), results)
                with col2:
                    st.subheader("检测结果")
                    st.image(result_img, use_container_width=True)
                
                # 统计
                st.metric("检测到人脸数", results['count'])
                
                # 详细信息
                if results['count'] > 0:
                    with st.expander("查看详情"):
                        for i in range(results['count']):
                            st.write(f"**人脸{i+1}**: 位置{results['locations'][i]}, 识别结果: {results['names'][i] or '未知'}")


if __name__ == "__main__":
    main()
