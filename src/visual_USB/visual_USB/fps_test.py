import cv2
import time
import numpy as np
from collections import deque

class FPSCounter:
    def __init__(self, max_frames=30):
        self.frame_times = deque(maxlen=max_frames)
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
    
    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        self.frame_count += 1
        
        # 計算最近30幀的平均幀率
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.fps = len(self.frame_times) / time_diff
        
        return self.fps
    
    def get_avg_fps(self):
        if self.frame_count > 0:
            total_time = time.time() - self.start_time
            return self.frame_count / total_time
        return 0.0

def test_camera_fps(camera_id=2, test_duration=10):
    """
    測試攝像頭幀率
    
    Args:
        camera_id: 攝像頭ID (默認2)
        test_duration: 測試持續時間(秒)
    """
    print(f"開始測試攝像頭 {camera_id} 的幀率...")
    print(f"測試持續時間: {test_duration} 秒")
    
    # 初始化攝像頭
    cap = cv2.VideoCapture(camera_id)
    
    
    if not cap.isOpened():
        print(f"錯誤：無法打開攝像頭 {camera_id}")
        return
    
    # 獲取攝像頭屬性
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_FPS, 270)  # 尝试设置 60 fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH,1280))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT,720))
    
    print(f"攝像頭屬性:")
    print(f"  分辨率: {width}x{height}")
    print(f"  聲稱幀率: {actual_fps:.2f} FPS")
    
    # 嘗試設置60 FPS
    cap.set(cv2.CAP_PROP_FPS, 270)
    set_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  設置幀率: {set_fps:.2f} FPS")
    
    # 初始化幀率計數器
    fps_counter = FPSCounter()
    
    # 測試開始時間
    start_time = time.time()
    frame_count = 0
    
    print("\n開始幀率測試...")
    print("按 'q' 鍵退出測試")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("錯誤：無法讀取幀")
            break
        
        frame_count += 1
        current_fps = fps_counter.update()
        
        # 在畫面上顯示幀率信息
        cv2.putText(frame, f"Current FPS: {current_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame Count: {frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Camera ID: {camera_id}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 顯示測試時間
        elapsed_time = time.time() - start_time
        cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('FPS Test', frame)
        
        # 檢查退出條件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or elapsed_time >= test_duration:
            break
    
    # 計算最終結果
    total_time = time.time() - start_time
    avg_fps = fps_counter.get_avg_fps()
    
    # 清理資源
    cap.release()
    cv2.destroyAllWindows()
    
    # 輸出結果
    print("\n" + "="*50)
    print("幀率測試結果:")
    print(f"  總測試時間: {total_time:.2f} 秒")
    print(f"  總幀數: {frame_count}")
    print(f"  平均幀率: {avg_fps:.2f} FPS")
    print(f"  最後30幀平均幀率: {fps_counter.fps:.2f} FPS")
    print(f"  每幀平均時間: {1000/avg_fps:.2f} ms" if avg_fps > 0 else "  每幀平均時間: N/A")
    
    # 幀率評估
    if avg_fps >= 25:
        print("  幀率狀態: 優秀 (>= 25 FPS)")
    elif avg_fps >= 15:
        print("  幀率狀態: 良好 (15-25 FPS)")
    elif avg_fps >= 10:
        print("  幀率狀態: 一般 (10-15 FPS)")
    else:
        print("  幀率狀態: 較低 (< 10 FPS)")
    
    print("="*50)
    
    return avg_fps

def test_multiple_cameras():
    """測試多個攝像頭"""
    print("多攝像頭幀率測試")
    print("="*50)
    
    results = {}
    
    # 測試攝像頭 0, 1, 2
    for camera_id in [0, 1, 2]:
        print(f"\n測試攝像頭 {camera_id}...")
        try:
            fps = test_camera_fps(camera_id, test_duration=5)
            results[camera_id] = fps
        except Exception as e:
            print(f"攝像頭 {camera_id} 測試失敗: {e}")
            results[camera_id] = None
    
    # 輸出比較結果
    print("\n" + "="*50)
    print("多攝像頭比較結果:")
    for camera_id, fps in results.items():
        if fps is not None:
            print(f"  攝像頭 {camera_id}: {fps:.2f} FPS")
        else:
            print(f"  攝像頭 {camera_id}: 無法訪問")
    
    # 找出最佳攝像頭
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_camera = max(valid_results, key=valid_results.get)
        best_fps = valid_results[best_camera]
        print(f"\n最佳攝像頭: 攝像頭 {best_camera} ({best_fps:.2f} FPS)")
    else:
        print("\n沒有可用的攝像頭")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--multi":
            test_multiple_cameras()
        else:
            try:
                camera_id = int(sys.argv[1])
                test_camera_fps(camera_id)
            except ValueError:
                print("用法: python fps_test.py [camera_id] 或 python fps_test.py --multi")
    else:
        # 默認測試攝像頭 2
        test_camera_fps(2) 