import cv2
import numpy as np

def mask_high_frequency_points(image_path, points):
    # 加載圖片
    image = cv2.imread(image_path, 0)

    # 執行傅立葉變換
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    # 創建與圖片大小相同的遮罩，初始值為1（白色）
    mask = np.ones(image.shape, np.uint8)
    
    # 在遮罩上將指定的高頻點標記為0（黑色）
    for point in points:
        mask[point[1], point[0]] = 0
    
    # 將遮罩應用於傅立葉變換後的頻譜
    fshift_masked = fshift * mask
    
    # 執行逆傅立葉變換
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # 保存結果
    result_path = 'masked_image.png'
    cv2.imwrite(result_path, img_back)
    
    return result_path

# 定義要遮蔽的點
points = [[321, 109], [238, 171], [404, 171], [155, 236], [486, 236], [238, 297], [404, 298], [321, 359]]

# 使用範例
image_path = 'UploadedImage3.jpg'  # 替換為你的圖片路徑
result_path = mask_high_frequency_points(image_path, points)
print(f"遮蔽後的圖片已保存為 {result_path}.")
