import cv2
import numpy as np
import pyrealsense2 as rs

# 影像寬高
w = 640
h = 480

# 定義 realsense camera
pipeline = rs.pipeline()
conf = rs.config()
# 設定彩色影像參數
conf.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
# 設定深度圖參數
conf.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)


# 開始影像串流
pipeline.start(conf)
cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
try:
    while True:
        # 等待並取得下一個frame的影像
        frames = pipeline.wait_for_frames()
        rgb_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not depth_frame or not rgb_frame:
            continue
        
        # 將影像轉為 cv2 格式
        rgb_image = np.asanyarray(rgb_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 合併兩張影像
        images = np.hstack((rgb_image,depth_colormap))
        # 顯示影像
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)

        # 按 esc or 'q' 來關閉視窗
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    # 停止影像串流
    pipeline.stop()