import cv2
import numpy as np
import matplotlib.pyplot as plt

def FourierTransform(image): # 進行傅立葉轉換
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 計算幅度譜
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    
    # 計算相位頻譜
    phase_spectrum = np.angle(dft_shift[:, :, 0] + 1j * dft_shift[:, :, 1])

    return dft_shift, magnitude_spectrum, phase_spectrum

def alterFourierTransform(f_shift): # 進行反傅立葉轉換
    f_ishift = np.fft.ifftshift(f_shift)
    img_processed = cv2.idft(f_ishift)
    img_processed = cv2.magnitude(img_processed[:, :, 0], img_processed[:, :, 1])
    
    return img_processed

def createWindow(image = None, title = None): # 創建一個圖形視窗
    plt.figure(figsize=(10, 6)) # 創建一個圖形視窗
    
    if image is not None:
        plt.imshow(image, cmap='gray')
        
    if title is not None:
        plt.title(title)
    
    # 去除圖形標框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    
    plt.xticks([]), plt.yticks([]) # 隱藏坐標軸刻度

def selfBuiltFilter(f_shift, img_notch = None, radius = 20, mode = "notch"):
    rows, cols = f_shift.shape[:2]
    mask = np.ones((rows, cols, 2), np.float32)

    for (x, y) in img_notch:
        rr, cc = np.ogrid[:rows, :cols] # 創建網格坐標，rr和cc分別表示行和列的坐標
        dist = (rr - y) ** 2 + (cc - x) ** 2

        if mode == "notch":
            mask[dist <= radius ** 2] = 0
        elif mode == "gaussian":
            gaussian = 1 - np.exp(- (dist) / (2 * (radius ** 2)))
            mask[:, :, 0] *= gaussian
            mask[:, :, 1] *= gaussian

        # 若尚未手動標記對稱點，可打開以下註解自動處理對稱點
        # y_sym, x_sym = rows - y, cols - x
        # dist_sym = (rr - y_sym) ** 2 + (cc - x_sym) ** 2
        # if mode == "notch":
        #     mask[dist_sym <= radius ** 2] = 0
        # elif mode == "gaussian":
        #     gaussian_sym = 1 - np.exp(-(dist_sym) / (2 * (radius ** 2)))
        #     mask[:, :, 0] *= gaussian_sym
        #     mask[:, :, 1] *= gaussian_sym

    return f_shift * mask

def normalize_and_save(filename, image): # 將圖像正規化並保存
    """
    OpenCV 的 cv2.imwrite() 只支援 uint8 或 uint16。
    如果丟入的是 float32 類型或數值超出 0–255 範圍，會自動做轉換。
    """
    norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.uint8(norm_img)
    cv2.imwrite(filename, norm_img)

if __name__ == "__main__":
    
    num = input("Please enter the image number (1 or 2): ")
    img = cv2.imread('input_image/image'+ num + '.jpg', 0)  # 以灰度模式讀取圖像
    
    img_points = {
        "img1_point": (
            (321, 109),
            (238, 171), (404, 171),
            (155, 236), (486, 236),
            (238, 297), (404, 298),
            (321, 359)
        ),
        "img2_point": (
            (100, 65), (100, 200), (100, 330), (100, 465), (100, 600),
            (200, 135), (200, 265), (200, 395), (200, 525), (200, 665),
            (300, 65), (300, 200), (300, 330), (300, 465), (300, 600),
            (400, 135), (400, 265), (400, 395), (400, 525), (400, 665),
            (500, 65), (500, 200), (500, 465), (500, 600),
            (600, 135), (600, 265), (600, 395), (600, 525), (600, 665),
            (700, 65), (700, 200), (700, 330), (700, 465), (700, 600),
            (800, 135), (800, 265), (800, 395), (800, 525), (800, 665),
            (900, 65), (900, 200), (900, 330), (900, 465), (900, 600),
            (1000, 135), (1000, 265), (1000, 395), (1000, 525), (1000, 665)
        )
    }
    key = "img" + num + "_point"
    
    if key in img_points:
        selected_points = img_points[key]
    else:
        print("Invalid image number.")
        exit(1)

    f_shift, magnitude_spectrum, phase_spectrum = FourierTransform(img)
    f_filtered = selfBuiltFilter(f_shift, img_notch=selected_points, radius=5, mode='notch') # 濾波器遮罩
    img_filtered = alterFourierTransform(f_filtered) # 進行反傅立葉轉換
    _, magnitude_spectrum_after, phase_spectrum_after = FourierTransform(img_filtered) # 重新計算濾波後的振幅頻譜和相位頻譜
    
    # createWindow(img)  # 顯示原圖
    # createWindow(magnitude_spectrum, "Magnitude Spectrum")  # 顯示原振幅頻譜
    # createWindow(phase_spectrum, "Phase Spectrum")  # 顯示原相位頻譜
    # createWindow(magnitude_spectrum_filtered, "Filtered Magnitude Spectrum")  # 顯示濾波後的振幅頻譜
    # createWindow(img_filtered, "Filtered Image")  # 顯示濾波後的圖像
    
    # 除錯用（可選）：顯示濾波器遮罩
    # mask = selfBuiltFilter(f_shift, img_notch=selected_points, radius=10, mode='notch')
    # createWindow(mask[:, :, 0], "Filter Mask")
    
    createWindow(title="Original Image")
    plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132).imshow(magnitude_spectrum, cmap='gray'), plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    plt.subplot(133).imshow(phase_spectrum, cmap='gray'), plt.title("Phase Spectrum"), plt.xticks([]), plt.yticks([])
    
    createWindow(title="Filtered Image")
    plt.subplot(131), plt.imshow(img_filtered, cmap='gray'), plt.title('Filtered Image'), plt.xticks([]), plt.yticks([]) 
    plt.subplot(132), plt.imshow(magnitude_spectrum_after, cmap='gray'), plt.title("Filtered Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(phase_spectrum_after, cmap='gray'), plt.title("Filtered Phase Spectrum"), plt.xticks([]), plt.yticks([])
    
    createWindow(title="Filtered Image Comparison")
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_filtered, cmap='gray'), plt.title("Filtered Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    
    # 顯示所有圖形窗口
    plt.show()
    
    # 將圖片存入output_image資料夾
    normalize_and_save('output_image/image' + num + '_magnitude_spectrum.jpg', magnitude_spectrum)
    normalize_and_save('output_image/image' + num + '_phase_spectrum.jpg', phase_spectrum)
    normalize_and_save('output_image/image' + num + '_filtered.jpg', img_filtered)
    normalize_and_save('output_image/image' + num + '_magnitude_spectrum_after.jpg', magnitude_spectrum_after)
    normalize_and_save('output_image/image' + num + '_phase_spectrum_after.jpg', phase_spectrum_after)
