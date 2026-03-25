import unittest
import numpy as np
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.face_processor import FaceProcessor


class TestFaceProcessor(unittest.TestCase):
    
    def test_init_empty(self):
        """测试空初始化"""
        p = FaceProcessor()
        self.assertEqual(len(p.known_names), 0)
    
    def test_detect_no_face(self):
        """测试无脸图片"""
        p = FaceProcessor()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = p.process(img)
        self.assertEqual(result['count'], 0)
    
    def test_recognize_no_library(self):
        """测试无人脸库"""
        p = FaceProcessor()
        fake_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = p.process(fake_img)
        self.assertEqual(result['names'], [])


if __name__ == '__main__':
    unittest.main()
