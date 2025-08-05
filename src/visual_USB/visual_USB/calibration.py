import cv2
import numpy as np
import glob

# 棋盘格参数
chessboard_size = (11, 8)  # 角点数目 (列, 行)，注意是内角点数
square_size = 20  # 每个方格的实际大小（单位：mm，可自定义）

# 终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 世界坐标系中的棋盘格点
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 存储所有图片的角点和三维点
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# 采集图片
cap = cv2.VideoCapture(4)  # 使用ZED立體攝像頭 (最佳幀率: 58.61 FPS)
# cap.set(cv2.CAP_PROP_EXPOSURE, 1000)  
cap.set(cv2.CAP_PROP_FPS, 60)  # 尝试设置 60 fps

img_count = 0
print("按空格拍照，按ESC退出采集。")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    frame = frame[:, :w//2]
    # 在圖像上現實帧率
    cv2.putText(frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC退出
        break
    elif key == 32:  # 空格拍照
        img_name = f'calib_img_{img_count}.jpg'
        cv2.imwrite(img_name, frame)
        print(f"已保存 {img_name}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()

# 标定
images = glob.glob('calib_img_*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # 可视化角点
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(200)
cv2.destroyAllWindows()

if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    print("相机内参矩阵：\n", mtx)
    print("畸变系数：\n", dist)
    # 计算重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    print(f"重投影误差: {mean_error}")
    # 保存参数
    np.savez('calib_result.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("标定结果已保存到 calib_result.npz")
else:
    print("未检测到足够的棋盘格角点，标定失败。")