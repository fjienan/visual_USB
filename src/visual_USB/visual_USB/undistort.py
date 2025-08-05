import cv2
import numpy as np
import argparse
import os
import glob
from pathlib import Path

class ImageUndistorter:
    def __init__(self, calib_file='calib_result.npz'):
        """
        初始化圖像去畸變器
        
        Args:
            calib_file: 標定參數文件路徑
        """
        self.calib_file = calib_file
        self.mtx = None
        self.dist = None
        self.newcameramtx = None
        self.roi = None
        
        # 加載標定參數
        self.load_calibration()
    
    def load_calibration(self):
        """加載相機標定參數"""
        try:
            with np.load(self.calib_file) as X:
                self.mtx = X['mtx']
                self.dist = X['dist']
            print(f"✅ 成功加載標定參數: {self.calib_file}")
            print(f"相機矩陣:\n{self.mtx}")
            print(f"畸變係數: {self.dist}")
        except FileNotFoundError:
            print(f"❌ 錯誤: 找不到標定文件 {self.calib_file}")
            print("請先運行 calibration.py 進行相機標定")
            return False
        except Exception as e:
            print(f"❌ 錯誤: 加載標定文件失敗 - {e}")
            return False
        return True
    
    def setup_undistortion(self, image_shape):
        """
        設置去畸變參數
        
        Args:
            image_shape: 圖像尺寸 (height, width)
        """
        if self.mtx is None or self.dist is None:
            return False
        
        h, w = image_shape[:2]
        # 計算優化後的新相機矩陣
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), 1, (w, h))
        
        print(f"圖像尺寸: {w}x{h}")
        print(f"ROI: {self.roi}")
        return True
    
    def undistort_image(self, image):
        """
        對單張圖像進行去畸變
        
        Args:
            image: 輸入圖像
            
        Returns:
            undistorted_image: 去畸變後的圖像
        """
        if self.mtx is None or self.dist is None:
            print("❌ 錯誤: 標定參數未加載")
            return image
        
        # 設置去畸變參數
        if not self.setup_undistortion(image.shape):
            return image
        
        # 進行去畸變
        undistorted = cv2.undistort(image, self.mtx, self.dist, None, self.newcameramtx)
        
        # 裁剪ROI區域
        if self.roi is not None:
            x, y, w, h = self.roi
            undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def process_single_image(self, input_path, output_path=None):
        """
        處理單張圖片
        
        Args:
            input_path: 輸入圖片路徑
            output_path: 輸出圖片路徑 (可選)
        """
        # 讀取圖片
        image = cv2.imread(input_path)
        if image is None:
            print(f"❌ 錯誤: 無法讀取圖片 {input_path}")
            return False
        
        print(f"處理圖片: {input_path}")
        
        # 去畸變
        undistorted = self.undistort_image(image)
        
        # 保存結果
        if output_path is None:
            name, ext = os.path.splitext(input_path)
            output_path = f"{name}_undistorted{ext}"
        
        cv2.imwrite(output_path, undistorted)
        print(f"✅ 已保存: {output_path}")
        
        # 顯示對比
        self.show_comparison(image, undistorted, input_path)
        
        return True
    
    def process_batch_images(self, input_dir, output_dir=None):
        """
        批量處理圖片
        
        Args:
            input_dir: 輸入目錄
            output_dir: 輸出目錄 (可選)
        """
        if output_dir is None:
            output_dir = input_dir + "_undistorted"
        
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找所有圖片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
        if not image_files:
            print(f"❌ 錯誤: 在目錄 {input_dir} 中找不到圖片文件")
            return False
        
        print(f"找到 {len(image_files)} 張圖片")
        
        success_count = 0
        for image_file in image_files:
            filename = os.path.basename(image_file)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_undistorted{ext}")
            
            if self.process_single_image(image_file, output_path):
                success_count += 1
        
        print(f"✅ 批量處理完成: {success_count}/{len(image_files)} 張圖片")
        return True
    
    def show_comparison(self, original, undistorted, title="Image Comparison"):
        """
        顯示原圖和去畸變圖的對比
        
        Args:
            original: 原圖
            undistorted: 去畸變圖
            title: 窗口標題
        """
        # 調整圖像大小以便並排顯示
        h, w = original.shape[:2]
        display_width = min(800, w)
        display_height = int(h * display_width / w)
        
        # 調整圖像大小
        original_resized = cv2.resize(original, (display_width, display_height))
        undistorted_resized = cv2.resize(undistorted, (display_width, display_height))
        
        # 並排顯示
        comparison = np.hstack([original_resized, undistorted_resized])
        
        # 添加標籤
        cv2.putText(comparison, "Original", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Undistorted", (display_width + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(title, comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def real_time_undistort(self, camera_id=4):
        """
        實時去畸變
        
        Args:
            camera_id: 攝像頭ID
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ 錯誤: 無法打開攝像頭 {camera_id}")
            return
        
        # 設置攝像頭參數
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("實時去畸變模式")
        print("按 'q' 退出, 's' 保存當前幀")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 錯誤: 無法讀取幀")
                break
            
            frame_count += 1
            
            # 去畸變
            undistorted = self.undistort_image(frame)
            
            # 顯示幀率
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(undistorted, f"Undistorted", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 並排顯示
            h, w = frame.shape[:2]
            display_width = min(800, w)
            display_height = int(h * display_width / w)
            
            frame_resized = cv2.resize(frame, (display_width, display_height))
            undistorted_resized = cv2.resize(undistorted, (display_width, display_height))
            comparison = np.hstack([frame_resized, undistorted_resized])
            
            cv2.imshow('Real-time Undistortion', comparison)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存當前幀
                cv2.imwrite(f'frame_{frame_count}_original.jpg', frame)
                cv2.imwrite(f'frame_{frame_count}_undistorted.jpg', undistorted)
                print(f"已保存幀 {frame_count}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='圖像去畸變工具')
    parser.add_argument('--calib', default='calib_result.npz', 
                       help='標定參數文件路徑 (默認: calib_result.npz)')
    parser.add_argument('--input', '-i', help='輸入圖片路徑')
    parser.add_argument('--output', '-o', help='輸出圖片路徑')
    parser.add_argument('--batch', help='批量處理目錄路徑')
    parser.add_argument('--realtime', '-r', action='store_true', 
                       help='實時去畸變模式')
    parser.add_argument('--camera', type=int, default=4, 
                       help='攝像頭ID (默認: 4)')
    
    args = parser.parse_args()
    
    # 創建去畸變器
    undistorter = ImageUndistorter(args.calib)
    
    if args.realtime:
        # 實時模式
        undistorter.real_time_undistort(args.camera)
    elif args.batch:
        # 批量處理模式
        undistorter.process_batch_images(args.batch, args.output)
    elif args.input:
        # 單張圖片模式
        undistorter.process_single_image(args.input, args.output)
    else:
        # 默認模式：處理標定圖片
        calib_images = glob.glob('calib_img_*.jpg')
        if calib_images:
            print("處理標定圖片...")
            for img_path in calib_images[:3]:  # 只處理前3張
                undistorter.process_single_image(img_path)
        else:
            print("用法:")
            print("  單張圖片: python undistort.py -i image.jpg")
            print("  批量處理: python undistort.py --batch input_dir")
            print("  實時模式: python undistort.py -r")
            print("  實時模式(指定攝像頭): python undistort.py -r --camera 0")

if __name__ == "__main__":
    main() 