# 人脸识别系统 (hw03)

基于 face_recognition (dlib) 和 Streamlit 的人脸检测与识别系统。

## 1. 项目结构
hw03/
├── app.py                 # Streamlit界面
├── src/                   # 核心代码
│   ├── init.py
│   └── face_processor.py  # 人脸检测/识别类
├── tests/                 # 测试
│   └── test_face.py
├── requirements.txt       # 依赖
├── README.md              # 本文件
└── known_faces/           # 已知人脸库（需自建）
└── 人名.jpg
## 2. 功能说明

**检测流程：**
1. 用户上传图片
2. 使用 face_recognition 检测人脸位置（HOG模型）
3. 提取128维人脸特征向量
4. 在图片上绘制绿框标注人脸

**识别流程：**
1. 加载 `known_faces/` 目录下的人脸库
2. 提取待识别人脸的128维特征
3. 计算与库中所有人脸的欧氏距离
4. 距离小于阈值则返回对应人名及置信度，否则显示"未知"

## 3. 如何准备人脸库

1. 在 `hw03/` 目录下创建 `known_faces/` 文件夹
2. 放入正面清晰的人脸照片（每人1张）
3. 图片命名格式：`人名.jpg`（如 `张三.jpg`、`Obama.jpg`）
4. 支持格式：jpg、jpeg、png

## 4. 运行与访问方式

**环境要求：**
- Python 3.8+
- 系统依赖：cmake（dlib需要，Windows一般已预装）

**安装依赖：**
```bash
pip install -r requirements.txt
