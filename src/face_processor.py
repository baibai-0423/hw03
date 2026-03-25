import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import os
from typing import List, Tuple, Optional, Dict


class FaceProcessor:
    """人脸处理：检测 + 128维特征 + 识别"""
    
    def __init__(self, known_faces_dir: str = None):
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        
        if known_faces_dir and os.path.exists(known_faces_dir):
            self._load_known_faces(known_faces_dir)
    
    def _load_known_faces(self, directory: str) -> None:
        """加载人脸库"""
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
                except Exception:
                    pass
    
    def process(self, image: np.ndarray, tolerance: float = 0.6) -> Dict:
        """检测+识别完整流程"""
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        face_names = []
        face_confidences = []
        
        for encoding in face_encodings:
            if self.known_encodings:
                distances = face_recognition.face_distance(self.known_encodings, encoding)
                best_idx = np.argmin(distances)
                best_dist = distances[best_idx]
                
                if best_dist <= tolerance:
                    face_names.append(self.known_names[best_idx])
                    face_confidences.append(round(1 - best_dist, 3))
                else:
                    face_names.append(None)
                    face_confidences.append(round(1 - best_dist, 3))
            else:
                face_names.append(None)
                face_confidences.append(None)
        
        return {
            'count': len(face_locations),
            'locations': face_locations,
            'names': face_names,
            'confidences': face_confidences,
            'encodings': face_encodings
        }
    
    def draw_results(self, image: Image.Image, results: Dict) -> Image.Image:
        """绘制检测结果"""
        draw = ImageDraw.Draw(image)
        for i, (top, right, bottom, left) in enumerate(results['locations']):
            draw.rectangle([left, top, right, bottom], outline="#00FF00", width=3)
            
            name = results['names'][i] or "未知"
            conf = results['confidences'][i]
            label = f"{name} ({conf:.1%})" if conf else name
            
            bbox = draw.textbbox((0, 0), label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.rectangle([left, top-text_h-10, left+text_w+10, top], fill="#00FF00")
            draw.text((left+5, top-text_h-5), label, fill="black")
        
        return image
