import cv2
from ultralytics import YOLO
from which_pose import classify_pose

def draw_simplified_pose(image, keypoints, confidence_threshold=0.3):
    """
    在视频帧中绘制简化版人体骨架，只关注头、肩、手臂、躯干和腿部。

    Args:
        image (np.array): 摄像头输入图像。
        keypoints (list): 关键点列表 [x, y, confidence]。
        confidence_threshold (float): 关键点可视化置信度阈值。
    """
    skeleton = [
        (5, 7), (7, 9),      # 左臂
        (6, 8), (8, 10),     # 右臂
        (11, 13), (13, 15),  # 左腿
        (12, 14), (14, 16),  # 右腿
        (5, 6),              # 肩连线
        (11, 12),            # 髋连线
        (5, 11), (6, 12),    # 肩到髋，构建躯干
        (0, 5), (0, 6)       # 鼻子到肩（头部连线）
    ]

    for person in keypoints:
        # 绘制关键点
        for i, keypoint in enumerate(person):
            x, y, conf = keypoint
            if conf > confidence_threshold:
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

        # 连接骨架
        for start_idx, end_idx in skeleton:
            if (person[start_idx][2] > confidence_threshold and person[end_idx][2] > confidence_threshold):
                x1, y1 = int(person[start_idx][0]), int(person[start_idx][1])
                x2, y2 = int(person[end_idx][0]), int(person[end_idx][1])
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)


def draw_custom_pose(image, keypoints, confidence_threshold=0.3):
    """
    绘制定制骨架：加入 neck 和 pelvis 两个中点，连接更合理。

    Args:
        image (np.array): 图像。
        keypoints (list): 每人 [17, 3] 的关键点列表。
        confidence_threshold (float): 最小置信度。
    """

    skeleton = [
        (1, 2),      # 左眼 - 右眼
        (0, 17),     # 鼻子 - neck
        (5, 7), (7, 9),    # 左臂
        (6, 8), (8, 10),   # 右臂
        (17, 5), (17, 6),  # neck - 肩
        (18, 11), (18, 12),# pelvis - 髋
        (11, 13), (13, 15),# 左腿
        (12, 14), (14, 16),# 右腿
        (17, 18)           # neck - pelvis
    ]

    for person in keypoints:
        keypts = person.tolist()

        # -- 构造 neck --
        if person[5][2] > confidence_threshold and person[6][2] > confidence_threshold:
            neck = [
                (person[5][0] + person[6][0]) / 2,
                (person[5][1] + person[6][1]) / 2,
                (person[5][2] + person[6][2]) / 2
            ]
        else:
            neck = [0, 0, 0]

        # -- 构造 pelvis --
        if person[11][2] > confidence_threshold and person[12][2] > confidence_threshold:
            pelvis = [
                (person[11][0] + person[12][0]) / 2,
                (person[11][1] + person[12][1]) / 2,
                (person[11][2] + person[12][2]) / 2
            ]
        else:
            pelvis = [0, 0, 0]

        # 添加两个中点到关键点列表（作为索引 17、18）
        keypts.append(neck)    # index 17
        keypts.append(pelvis)  # index 18

        # 绘制所有关键点
        for x, y, conf in keypts:
            if conf > confidence_threshold:
                cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)

        # 绘制骨架线
        for i, j in skeleton:
            if keypts[i][2] > confidence_threshold and keypts[j][2] > confidence_threshold:
                x1, y1 = int(keypts[i][0]), int(keypts[i][1])
                x2, y2 = int(keypts[j][0]), int(keypts[j][1])
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)


def real_time_pose_estimation(camera_index=0, model_path='yolov8n-pose.pt', confidence_threshold=0.3):
    """
    实时摄像头人体姿态检测。

    Args:
        camera_index (int): 摄像头索引（默认 0）。
        model_path (str): YOLOv8-Pose 模型路径。
        confidence_threshold (float): 关键点置信度阈值。
    """
    print(f"🎥 正在打开摄像头索引 {camera_index}...")

    # 加载 YOLOv8-Pose 模型
    model = YOLO(model_path)
    print(f"✅ 成功加载模型: {model_path}")

    # 打开摄像头
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ 错误: 无法打开摄像头！")
        return

    print("🚀 按 'q' 退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 错误: 读取摄像头帧失败！")
            break

        # 检测人体姿态
        results = model(frame, conf=confidence_threshold)
        keypoints = results[0].keypoints.data.cpu().numpy()  # 提取关键点

        # 绘制火柴人骨架
        draw_custom_pose(frame, keypoints)
        # 动作识别：调用 which_pose.classify_pose
        for person in keypoints:
            action_label = classify_pose(person)

            # 在鼻子处绘制文字（关键点索引 0）
            nose = person[0]
            if nose[2] > confidence_threshold:
                cv2.putText(frame, action_label, (int(nose[0]), int(nose[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # 显示结果
        cv2.imshow("实时姿态检测", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 退出程序，摄像头已释放。")


if __name__ == "__main__":
    real_time_pose_estimation()
