import cv2
import numpy as np
import pickle
import sys
sys.path.append('D:\AIFace\_internal\DeepFaceLab\DFLIMG\DFLJPG.py')
from DFLJPG import DFLJPG
# 假设 DFLJPG 类已经在你的脚本中定义了

def extract_face_info_from_dfl(filename):
    # 使用 DFLJPG 类加载图片
    dfl_img = DFLJPG.load(filename)

    # 检查是否成功加载了 dfl_dict
    if not dfl_img.dfl_dict:
        raise ValueError("DFL dictionary is missing in the image")

    # 获取人脸相关信息
    face_info = dfl_img.dfl_dict.get('face_info', None)
    if face_info is None:
        raise ValueError("Face information is missing in the DFL dictionary")

    # 解析人脸区域和角度
    face_bbox = face_info.get('face_bbox', None)  # 人脸边界框
    pitch_yaw_roll = face_info.get('pitch_yaw_roll', None)  # 人脸角度

    return face_bbox, pitch_yaw_roll

def display_face_info(filename):
    try:
        face_bbox, pitch_yaw_roll = extract_face_info_from_dfl(filename)
        print(f"Face bounding box: {face_bbox}")
        print(f"Pitch, Yaw, Roll: {pitch_yaw_roll}")
        
        # 可选: 显示图像和人脸边界框
        img = cv2.imread(filename)
        if img is not None and face_bbox is not None:
            x, y, w, h = face_bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Face Detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # 替换为你的DFL图片文件路径
    filename = r'D:\AIFace\JM face\data_src\aligned\2759.jpg'
    display_face_info(filename)
