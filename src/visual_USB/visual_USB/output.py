# ...existing code...
import cv2
import numpy as np

# 加載標定參數
with np.load('calib_result.npz') as X:
    mtx, dist = X['mtx'], X['dist']
# 輸出相機矩陣和畸變係數
print(f"相機矩陣: {mtx}")
print(f"畸變係數: {dist}")
# 讀取一張待去畸變的圖片
img = cv2.imread('calib_img_1.jpg')
h, w = img.shape[:2]

# 計算優化後的新相機矩陣
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 去畸變
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# 顯示結果
cv2.imshow('undistorted', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()