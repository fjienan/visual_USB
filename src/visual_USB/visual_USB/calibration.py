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
cap = cv2.VideoCapture(2)  # 使用ZED立體攝像頭 (最佳幀率: 58.61 FPS)
if not cap.isOpened():
    print("错误：无法打开摄像头")
    exit()

# --- 关键步骤：设置像素格式为 MJPEG ---
# 使用 fourcc 编码 'M', 'J', 'P', 'G'
# 注意：必须在设置分辨率和帧率之前设置 FOURCC
fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
set_fourcc = cap.set(cv2.CAP_PROP_FOURCC, fourcc)
if not set_fourcc:
    print("警告：设置 MJPG 格式失败，你的摄像头可能不支持或者 V4L2 后端有问题。")
    # 可以尝试不设置 FOURCC，看看默认是什么格式
    # current_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # print(f"当前 FOURCC: {chr(current_fourcc & 0xff)}{chr((current_fourcc >> 8) & 0xff)}{chr((current_fourcc >> 16) & 0xff)}{chr((current_fourcc >> 24) & 0xff)}")

# --- 设置期望的分辨率 ---
# 比如设置 1920x1080
width = 1280
height = 1024
set_width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
set_height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
if not set_width or not set_height:
    print(f"警告：设置分辨率 {width}x{height} 失败。")

# --- 设置期望的帧率 ---
# 尝试设置 30 FPS
fps = 190.0
set_fps = cap.set(cv2.CAP_PROP_FPS, fps) # 使用浮点数
if not set_fps:
    print("警告：设置 30 FPS 失败。驱动可能会返回一个它能支持的帧率。")

# 检查实际生效的参数
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
actual_fourcc_str = f"{chr(actual_fourcc_int & 0xff)}{chr((actual_fourcc_int >> 8) & 0xff)}{chr((actual_fourcc_int >> 16) & 0xff)}{chr((actual_fourcc_int >> 24) & 0xff)}"

print(f"摄像头已打开。")
print(f"请求参数：Format=MJPG, Width={width}, Height={height}, FPS={fps:.2f}")
print(f"实际参数：Format={actual_fourcc_str}, Width={int(actual_width)}, Height={int(actual_height)}, FPS={actual_fps:.2f}")
img_count = 0
print("按空格拍照，按ESC退出采集。")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape    # 在圖像上現實帧率
    # cv2.putText(frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS)}", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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