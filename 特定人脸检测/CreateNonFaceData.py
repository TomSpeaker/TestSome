import cv2
import os

# 输入视频文件路径
video_path = 'vedioback.mp4'  # 替换为你的视频路径
cap = cv2.VideoCapture(video_path)

# 创建输出文件夹
output_folder = 'background_cuts_from_video'
os.makedirs(output_folder, exist_ok=True)

# ✅ 设置起始编号和目标数量
start_index = 0  # 从 cut_501.jpg 开始编号
target_count = 300  # 要保存的图像总数

frame_count = 0
cut_count = 0

while cap.isOpened() and cut_count < target_count:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    for y in range(0, height, 50):
        for x in range(0, width, 50):
            if x + 50 <= width and y + 50 <= height:
                cut_img = frame[y:y+50, x:x+50]

                # 保存图像，编号从 start_index 开始
                cut_filename = os.path.join(output_folder, f"cut_{start_index + cut_count}.jpg")
                cv2.imwrite(cut_filename, cut_img)
                cut_count += 1

                if cut_count >= target_count:
                    break
        if cut_count >= target_count:
            break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"裁剪完成，共生成 {cut_count} 张图像，编号从 cut_{start_index}.jpg 开始，保存至文件夹：{output_folder}")
