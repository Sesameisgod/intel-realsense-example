# intel-realsense-example
### **建置環境**
- 將專案clone到電腦上並進入到intel-realsense-example/，︀輸入`pip install -r requirements.txt`安裝對應套件
### **操作說明**
- 讀取單支 realsense 攝影機影像
  1. 將 realsense 攝影機插入電腦
  2. 執行 `python read_cam.py`
- 讀取多支 realsense 攝影機影像
  1. 將全部要用到的 realsense 攝影機插入電腦
  2. 進入 read_multiple_cams.py 中，將所有攝影機的S/N碼輸入進`sn_number_list`中 (相機的S/N碼可透過執行`python get_SN_number.py`來取得)
- 3D 投影演示
  1. 將 realsense 攝影機插入電腦
  2. 執行 `python 3d_projection.py`
  3. 畫面中左右手腕旁的紅色文字分別對應其與相機的三維空間距離
### **連結**
- [pyrealsense2(realsense python sdk) document](https://github.com/IntelRealSense/librealsense/tree/master/doc)
- [mediapipe document](https://developers.google.com/mediapipe)
