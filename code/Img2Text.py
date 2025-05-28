import easyocr
from PIL import Image
import numpy as np
import cv2


def image_to_text(image_path, languages=None):
    try:
        # 尝试用 PIL 打开图片
        pil_image = Image.open(image_path)
        # 将 PIL 图像转换为 OpenCV 格式
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 创建 reader 对象
        if languages is None:
            languages = ['ch_sim', 'en']
        reader = easyocr.Reader(languages)

        # 读取图片并识别文字
        result = reader.readtext(cv_image)

        # 提取文本
        text = '\n'.join([item[1] for item in result])

        return text
    except Exception as e:
        print(f"Error reading image: {e}")
        return ''
