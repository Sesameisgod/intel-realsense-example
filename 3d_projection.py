import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from time import time

# 全身關節點偵測網路(demo 用)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

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
# 影像對齊
align_to = rs.stream.color      # align_to 是计划对齐深度帧的流类型
align = rs.align(align_to)

# 相機內參
color_intrin = None
# 深度內參
depth_intrin = None


# 開始影像串流
pipeline.start(conf)
cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
try:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            start_time = time()
            # 等待並取得下一個frame的影像
            frames = pipeline.wait_for_frames()

            # 對齊影像
            aligned_frames = align.process(frames)
            rgb_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            # 獲取內參
            color_intrin = rgb_frame.profile.as_video_stream_profile().intrinsics
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            if not depth_frame or not rgb_frame:
                continue
            
            # 將影像轉為 cv2 格式
            rgb_image = np.asanyarray(rgb_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 偵測關節點
            # 將影像設為 not writeable 來加速運算 (可選)
            rgb_image.flags.writeable = False
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_image)

            # 畫出關節點
            rgb_image.flags.writeable = True
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                rgb_image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                rgb_image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
            
            if results:
                # 取得 左手關節點
                if results.left_hand_landmarks:
                    landmarks = results.left_hand_landmarks.landmark
                    for i in landmarks:
                        x = int(i.x * w)
                        y = int(i.y * h)
                        cv2.circle(rgb_image, (x, y), 5, (0, 255, 0), -1)
                
                # 取得 右手關節點
                if results.right_hand_landmarks:
                    landmarks = results.right_hand_landmarks.landmark
                    for i in landmarks:
                        x = int(i.x * w)
                        y = int(i.y * h)
                        cv2.circle(rgb_image, (x, y), 5, (0, 255, 0), -1)

                # 取得 pose關節點
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    pose = []
                    for i in landmarks:
                        x = int(i.x * w)
                        y = int(i.y * h)
                        pose.append([x,y])
                    # 取得手腕關節點
                    left_wrist = pose[15]
                    right_wrist = pose[16]

                    # 取得 3d 投影點座標 (目標 pixel 與相機的相對位置)
                    for coordinage_2d in [left_wrist,right_wrist]:
                        x,y = coordinage_2d
                        if x > 0 and y > 0:
                            dis = depth_frame.get_distance(x, y)
                            coordinage_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, coordinage_2d, dis)
                            # 畫出 3d 座標 (單位 : 公分)
                            cv2.putText(rgb_image, f"({round(coordinage_3d[0]*100)},{round(coordinage_3d[1]*100)},{round(coordinage_3d[2]*100)})", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, cv2.LINE_AA)


            # 計算並顯示 FPS
            cv2.putText(depth_colormap, f"FPS={round(1/(time()-start_time))}", (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1, cv2.LINE_AA)
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