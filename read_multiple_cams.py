import cv2
import numpy as np
import pyrealsense2 as rs

# 【讀取多支攝影機】

# 影像寬高
w = 640
h = 480

# 指定攝影機 SN 碼
sn_number_list = ['927522071726','925322061507']

pipeline_list = []
conf_list = []

# 定義 realsense camera
for i,sn_number in enumerate(sn_number_list):
    pipeline_list.append(rs.pipeline())
    conf_list.append(rs.config())
    conf_list[i].enable_device(sn_number)
    # 設定彩色影像參數
    conf_list[i].enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
    # 設定深度圖參數
    conf_list[i].enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)

# 開始影像串流
[pipeline.start(conf) for pipeline,conf in zip(pipeline_list,conf_list)]

cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
try:
    while True:
        # 等待並取得下一個frame的影像
        frames_list = [pipeline.wait_for_frames() for pipeline in pipeline_list]
        rgb_frame_list = [frames.get_color_frame() for frames in frames_list]
        depth_frame_list = [frames.get_depth_frame() for frames in frames_list]
        # 將影像轉為 cv2 格式
        rgb_image_list = [np.asanyarray(rgb_frame.get_data()) for rgb_frame in rgb_frame_list]
        depth_image_list = [np.asanyarray(depth_frame.get_data()) for depth_frame in depth_frame_list]
        depth_colormap_list = [cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) for depth_image in depth_image_list]
        # 合併影像
        rgb_image = np.vstack(rgb_image_list)
        depth_image = np.vstack(depth_colormap_list)
        images = np.hstack((rgb_image,depth_image))
        # 顯示影像
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)

        # 按 esc or 'q' 來關閉視窗
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    # 停止影像串流
    [pipeline.stop() for pipeline in pipeline_list]