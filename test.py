import cv2
import numpy as np

def apply_black_dot_filter(image_path, bright_spots):
    # 加載圖片
    image = cv2.imread(image_path, 0)
    
    # 執行FFT以獲取頻域表示
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    # 創建與頻域表示相同大小的遮罩
    rows, cols = image.shape
    mask = np.ones((rows, cols), np.uint8)
    
    # 在頻域中對亮點應用黑點
    for spot in bright_spots:
        mask[spot[1], spot[0]] = 0
    
    # 將遮罩應用於頻域表示
    fshift_filtered = fshift * mask
    
    # 執行逆FFT以獲取過濾後的圖片
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # 保存過濾後的圖片
    filtered_image_path = 'filtered_image.png'
    cv2.imwrite(filtered_image_path, img_back)
    
    return filtered_image_path
# 使用者提供的亮點座標
bright_spots = [[321, 109], [238, 171], [404, 171], [155, 236], [486, 236], [238, 297], [404, 298], [321, 359]]

# 將過濾器應用於圖片（請將'input_image.png'替換為您的圖片文件路徑）
filtered_image_path = apply_black_dot_filter('input_image/image1.jpg', bright_spots)

print(f"過濾後的圖片已保存為 {filtered_image_path}.")