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

def createWindow(image = None, title = None): # 創建一個圖形視窗
    plt.figure() # 創建一個圖形視窗
    
    if image is not None:
        plt.imshow(image, cmap='gray')

    if title is not None:
        plt.title(title)
    else:
        # 去除圖形標框
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        
    plt.xticks([]), plt.yticks([]) # 隱藏坐標軸刻度

def ideaFilter(img, sigma = 10, img_notch = None):
    img_filtered = img.copy()
    
    for point in img_notch:
        cv2.circle(img_filtered, tuple(point), radius = 5, color = 0, thickness = -1)

    return img_filtered

def alterFourierTransform(image):
    f_ishift = np.fft.ifftshift(image)
    img_processed = np.fft.ifft2(f_ishift)
    img_processed = np.abs(img_processed)
    
    return img_processed

if __name__ == "__main__":
    
    img = cv2.imread('input_image/image1.jpg', 0)  # 以灰度模式讀取圖像
    
    row, col = img.shape
    
    img_point = [[321, 109], 
                 [238, 171], [404, 171], 
                 [155, 236], [486, 236], 
                 [238, 297], [404, 298],
                 [321, 359]]

    magnitude_spectrum, phase_spectrum = FourierTransform(img)
    magnitude_spectrum_filtered = ideaFilter(magnitude_spectrum, sigma=10, img_notch = img_point)
    image_filtered = alterFourierTransform(magnitude_spectrum_filtered) # 進行反傅立葉轉換
    magnitude_spectrum_filtered, phase_spectrum_filtered = FourierTransform(image_filtered) # 重新計算濾波後的幅度譜和相位頻譜
    
    # createWindow(img)  # 顯示原圖
    # createWindow(magnitude_spectrum, "Magnitude Spectrum")  # 顯示原幅度譜
    # createWindow(phase_spectrum, "Phase Spectrum")  # 顯示原相位頻譜
    createWindow(magnitude_spectrum_filtered, "Filtered Magnitude Spectrum")  # 顯示濾波後的幅度譜
    createWindow(image_filtered, "Filtered Image")  # 顯示濾波後的圖像
    
    createWindow()
    plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132).imshow(magnitude_spectrum, cmap='gray'), plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    plt.subplot(133).imshow(phase_spectrum, cmap='gray'), plt.title("Phase Spectrum"), plt.xticks([]), plt.yticks([])
    
    # 顯示所有圖形窗口
    plt.show()
    
    # print("mask:{}".format(gaussianFilter(magnitude_spectrum, sigma=10, img_notch = img_point)))