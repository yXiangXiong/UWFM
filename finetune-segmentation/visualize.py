import numpy as np
from PIL import Image

def save_mask(tensor, file_path, num_class, original_image=None):
    """
    Save the mask as an RGB image or overlay it on the original image.
    
    Args:
        tensor: PyTorch tensor of shape [H, W], containing class indices
        file_path: str, path to save the PNG image
        num_class: int, number of classes to map
        original_image: PIL Image or None, if provided, overlay the mask on this image
    """
    arr = tensor.squeeze().cpu().numpy().astype(np.uint8)
    if num_class > 20:
        raise ValueError(f"Currently supports up to 20 classes, received {num_class} classes")

    # 定义 20 种颜色的映射 (RGB)
    colors = [
        (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128), (192, 192, 192), (255, 165, 0),
        (255, 192, 203), (75, 0, 130), (173, 216, 230), (240, 230, 140), (139, 69, 19)
    ]
    
    # 生成彩色掩码
    arr_rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    for cls in range(num_class):
        arr_rgb[arr == cls] = colors[cls]
    
    # 如果提供了原始图像，则叠加掩码
    if original_image is not None:
        # 将原始图像转换为 numpy 数组
        orig_arr = np.array(original_image)
        # 确保尺寸匹配
        if orig_arr.shape[:2] != arr.shape:
            orig_arr = np.array(original_image.resize((arr.shape[1], arr.shape[0]), Image.BILINEAR))
        
        # 叠加掩码（半透明效果）
        alpha = 0.5  # 掩码透明度
        overlay = (orig_arr * (1 - alpha) + arr_rgb * alpha).astype(np.uint8)
        image = Image.fromarray(overlay, mode='RGB')
    else:
        image = Image.fromarray(arr_rgb, mode='RGB')
    
    image.save(file_path)