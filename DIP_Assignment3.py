import cv2
import numpy as np
import matplotlib.pyplot as plt

def FourierTransform(image):
    # 進行傅立葉轉換
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 計算幅度譜
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    
    # 計算相位頻譜
    phase_spectrum = np.angle(dft_shift[:, :, 0] + 1j * dft_shift[:, :, 1])

    return magnitude_spectrum, phase_spectrum

def createWindow(image, title = 'Image'): # 創建一個圖形窗口並顯示原圖
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])

def gaussianFilter(image, sigma = 10, img_notch = None):
    # 設置高斯濾波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    sigma = 30
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    x, y = np.meshgrid(x, y)
    mask = np.exp(-(x**2 + y**2) / (2 * sigma**2)) # 高斯函數
    mask = mask / mask.max()
    mask = np.repeat(mask[:, :, np.newaxis], 2, axis = 2)  # 重複以匹配通道數

    # filtered_image = image * mask

    # return filtered_image

if __name__ == "__main__":
    
    img = cv2.imread('input_image/image1.jpg', 0)  # 以灰度模式讀取圖像
    
    row, col = img.shape
    img_point = [[321, 109], 
                 [238, 171], [404, 171], 
                 [155, 236], [486, 236], 
                 [238, 297], [404, 298],
                 [321, 359]]

    magnitude_spectrum, phase_spectrum = FourierTransform(img)
    # filtered_image = gaussianFilter(magnitude_spectrum, sigma=10, img_notch = img_point)
    
    createWindow(img)  # 顯示原圖
    createWindow(magnitude_spectrum, "Magnitude Spectrum")  # 顯示幅度譜
    createWindow(phase_spectrum, "Phase Spectrum")  # 顯示相位頻譜
    # createWindow(filtered_image, "Filtered Image")  # 顯示濾波後的圖像

    # 顯示所有圖形窗口
    plt.show()