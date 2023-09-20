import pyrealsense2 as rs
import numpy as np
import cv2

# 取得目前連接的所有相機資訊

ctx = rs.context()
devices = ctx.query_devices()

for device in devices:
    print(device)