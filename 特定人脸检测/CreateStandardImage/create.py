import cv2
import time

# 加载 OpenCV 人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 打开默认摄像头
cap = cv2.VideoCapture(0)

# 获取摄像头图像的宽高
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 设置输出视频参数（不带红框，MP4格式）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 改为 'mp4v' 输出 MP4 格式
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# 定义红色矩形框大小（以画面中心为中心）
box_width, box_height = 400, 400

# 设置录制时长（秒）
record_time = 60  # 

# 获取当前时间（开始时间）
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 获取当前时间并计算已录制时长
    elapsed_time = time.time() - start_time
    remaining_time = int(record_time - elapsed_time)  # 计算剩余时间（秒）
    
    if remaining_time <= 0:
        break  # 超过录制时长后退出

    # 转为灰度图像用于人脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用人脸检测器检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 保存不带框的视频
    out.write(frame.copy())

    # ===== 显示用帧（加红框和人脸检测框） =====
    display_frame = frame.copy()

    # 画红色矩形框（居中）
    center_x, center_y = frame_width // 2, frame_height // 2
    x1 = center_x - box_width // 2
    y1 = center_y - box_height // 2
    x2 = center_x + box_width // 2
    y2 = center_y + box_height // 2
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色矩形框

    # 在每一张检测到人脸的区域画绿色矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色矩形框

    # 显示剩余时间
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display_frame, f'Remaining: {remaining_time}s', (10, 30), font, 1, (0, 255, 0), 2)

    # 显示视频窗口
    cv2.imshow('Camera with Red Box and Face Detection', display_frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"录制完成，已保存为 output.mp4")
