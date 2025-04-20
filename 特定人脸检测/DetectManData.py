import cv2
import os

# 加载 OpenCV 人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 输入视频文件路径
video_path = 'output.mp4'  # 你的视频路径
cap = cv2.VideoCapture(video_path)

# 创建输出文件夹
output_folder = 'detected_faces'
os.makedirs(output_folder, exist_ok=True)

# 设置起始编号和目标保存数量
start_index = 0  # ✅ 从第几张图片开始命名
target_face_count = 300  # ✅ 想保存多少张人脸

frame_count = 0
saved_count = 0

while cap.isOpened() and saved_count < target_face_count:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for i, (fx, fy, fw, fh) in enumerate(faces):
        if fw >200 and fh > 200:
            face_img = frame[fy:fy+fh, fx:fx+fw]
            fixed_size_face = cv2.resize(face_img, (50, 50))

            # 使用 start_index 生成文件名
            face_filename = os.path.join(output_folder, f"face_{start_index + saved_count}.jpg")
            cv2.imwrite(face_filename, fixed_size_face)
            saved_count += 1

            # 达到目标数量后退出
            if saved_count >= target_face_count:
                break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"共保存了 {saved_count} 张人脸图像，保存至文件夹：{output_folder}，编号从 face_{start_index}.jpg 开始")
