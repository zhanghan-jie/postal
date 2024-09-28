#
# # import cv2
# # import numpy as np
# # import torch
# # from models.experimental import attempt_load
# # from utils.general import check_img_size, non_max_suppression, scale_coords
# # # from utils.plots import plot_one_box
# # from utils.torch_utils import select_device
# #
# #
# # def load_model(weights='yolov5s.pt'):
# #     device = select_device('')  # 默认使用 CPU，如果有 GPU 则使用 GPU
# #     model = attempt_load(weights, device=device)  # load FP32 model
# #     return model
# #
# #
# # def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
# #     """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
# #     shape = im.shape[:2]  # current shape [height, width]
# #     if isinstance(new_shape, int):
# #         new_shape = (new_shape, new_shape)
# #
# #     # Scale ratio (new / old)
# #     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
# #     if not scaleup:  # only scale down, do not scale up (for better val mAP)
# #         r = min(r, 1.0)
# #
# #     # Compute padding
# #     ratio = r, r  # width, height ratios
# #     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
# #     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
# #     if auto:  # minimum rectangle
# #         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
# #     elif scaleFill:  # stretch
# #         dw, dh = 0.0, 0.0
# #         new_unpad = (new_shape[1], new_shape[0])
# #         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
# #
# #     dw /= 2  # divide padding into 2 sides
# #     dh /= 2
# #
# #     if shape[::-1] != new_unpad:  # resize
# #         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
# #     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
# #     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
# #     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
# #     return im, ratio, (dw, dh)
# #
# #
# # def detect_objects(model, source, img_size=640, conf_thres=0.25, iou_thres=0.45):
# #     img0 = cv2.imread(source)  # BGR
# #     assert img0 is not None, 'Image Not Found ' + source
# #
# #     # Resize and pad image
# #     img, ratio, (dw, dh) = letterbox(img0, new_shape=img_size)
# #
# #     # Convert
# #     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
# #     img = np.ascontiguousarray(img)
# #
# #     # Convert to torch tensor
# #     device = next(model.parameters()).device  # 获取模型所在的设备
# #     img = torch.from_numpy(img).to(device)
# #     img = img.float()  # uint8 to fp32
# #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
# #     if img.ndimension() == 3:
# #         img = img.unsqueeze(0)  # add batch dimension
# #
# #     # Inference
# #     pred = model(img, augment=False)[0]
# #
# #     # Apply NMS
# #     pred = non_max_suppression(pred, conf_thres, iou_thres)
# #
# #     # Process detections
# #     detections = []
# #     for i, det in enumerate(pred):  # detections per image
# #         gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
# #         if det is not None and len(det):
# #             # Rescale boxes from img_size to im0 size
# #             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
# #             detections.append(det.cpu())  # Move to CPU
# #     return detections
# #
# #
# # def measure_object_size(image, bbox):
# #     x1, y1, x2, y2 = bbox
# #     cropped_image = image[y1:y2, x1:x2]
# #     gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
# #     edges = cv2.Canny(gray, 50, 150)
# #     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     largest_contour = max(contours, key=cv2.contourArea)
# #     rect = cv2.minAreaRect(largest_contour)
# #     width, height = rect[1]
# #     return width, height
# #
# #
# # def main():
# #     weights = r'E:\8.8\yolov5-master\runs\train\exp77\weights\best.pt'  # or your custom model path
# #     source = r'F:\image\20220923063125.jpg'
# #
# #     model = load_model(weights)
# #     detections = detect_objects(model, source)
# #
# #     if detections:
# #         img = cv2.imread(source)
# #         for det in detections:
# #             for *xyxy, conf, cls in det:
# #                 label = f'{model.names[int(cls)]} {conf:.2f}'
# #                 xyxy = torch.tensor(xyxy).view(-1).tolist()  # Convert to list
# #                 save_one_box(xyxy, img, label=label, color=[255, 0, 0], line_thickness=3)
# #                 width, height = measure_object_size(img, xyxy)
# #                 print(f"包裹的宽度为：{width} 像素, 高度为：{height} 像素")
# #         cv2.imshow('Detection', img)
# #         cv2.waitKey(0)
# #         cv2.destroyAllWindows()
# #     else:
# #         print("没有检测到任何物体。")
# #
# #
# # if __name__ == '__main__':
# #     main()
#
#
# import numpy as np
# import cv2
# import torch
# from ultralytics import YOLO
#
# # 定义比例尺（假设每像素代表的实际长度）
# scale = 0.01  # 这里假设100像素代表1厘米
#
# # 加载YOLOv5模型
# model = YOLO('path/to/your/best.pt')  # 使用正确的路径替换
#
#
# # 定义函数来检测包裹并计算尺寸
# def detect_and_measure(frame, conf_threshold=0.1):
#     results = model(frame)[0]
#
#     # 提取边界框信息
#     bboxes = results.boxes.xyxy.cpu().numpy()
#     confidences = results.boxes.conf.cpu().numpy()
#     classes = results.boxes.cls.cpu().numpy()
#
#     # 筛选出高置信度的检测结果
#     valid_indices = confidences > conf_threshold
#     valid_bboxes = bboxes[valid_indices]
#     valid_classes = classes[valid_indices]
#
#     # 计算包裹的实际尺寸
#     parcel_dimensions = calculate_dimensions(valid_bboxes)
#
#     return parcel_dimensions, valid_bboxes, valid_classes
#
#
# # 定义函数来计算每个包裹的尺寸
# def calculate_dimensions(bboxes):
#     dimensions = []
#     for bbox in bboxes:
#         # bbox格式为 [xmin, ymin, xmax, ymax]
#         x1, y1, x2, y2 = bbox
#
#         # 确保 x2 >= x1 和 y2 >= y1
#         if x2 <= x1 or y2 <= y1:
#             print(f"Invalid bounding box coordinates: {bbox}")
#             continue
#
#         width_pixels = int(x2 - x1)
#         height_pixels = int(y2 - y1)
#
#         width_cm = width_pixels * scale
#         height_cm = height_pixels * scale
#
#         dimensions.append((width_cm, height_cm))
#
#     return dimensions
#
#
# # 初始化摄像头或其他视频源
# video_capture = cv2.VideoCapture(0)  # 使用0表示默认摄像头，也可以使用视频文件路径
#
# while True:
#     ret, frame = video_capture.read()
#     if not ret:
#         break
#
#     # 检测包裹并计算尺寸
#     parcel_dimensions, valid_bboxes, valid_classes = detect_and_measure(frame, conf_threshold=0.1)
#
#     # 在每一帧上绘制边界框
#     for bbox, dim in zip(valid_bboxes, parcel_dimensions):
#         x1, y1, x2, y2 = bbox.astype(int)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"W: {dim[0]:.2f}cm, H: {dim[1]:.2f}cm"
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#     # 显示带有标注的视频流
#     cv2.imshow('Parcel Detection', frame)
#
#     # 按 'q' 键退出循环
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 释放视频捕获设备
# video_capture.release()
# cv2.destroyAllWindows()



import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.utils.plotting import save_one_box  # 假设save_one_box位于此模块中
from pathlib import Path


def load_model(model_path):
    # 加载模型权重
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.eval()  # 设置模型为评估模式
    return model


def draw_boxes_and_calculate_dimensions(frame, model, save_crops=True):
    # 将 OpenCV 图像转换为 PIL 图像
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 使用模型进行推理
    results = model(pil_image, size=640)  # 设置图像大小

    # 获取检测结果
    predictions = results.xyxy[0].tolist()

    # 创建一个新的图像绘图对象
    frame_copy = frame.copy()

    # 遍历每一个检测结果
    for idx, pred in enumerate(predictions):
        # pred 格式为 [x1, y1, x2, y2, confidence, class_id]
        x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
        confidence = pred[4]
        class_id = int(pred[5])

        # 在帧上绘制边界框
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 计算边界框的宽度和高度
        width = x2 - x1
        height = y2 - y1

        # 构造标签字符串
        label = (
            f"Class ID: {class_id}, Confidence: {confidence:.2f}, "
            f"Coordinates: (x1, y1, x2, y2) = ({x1}, {y1}, {x2}, {y2}), "
            f"Width: {width} pixels, Height: {height} pixels"
        )

        # 在边界框上方写入类别ID、置信度、坐标、宽度和高度
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = max(x1, frame.shape[1] - text_size[0] - 10)
        text_y = max(y1 - 10, text_size[1] + 10)

        # 绘制背景矩形以提高可读性
        cv2.rectangle(frame_copy, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0, 255, 0), -1)
        cv2.putText(frame_copy, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 打印坐标、置信度、宽度和高度
        print(label)

        # # 保存裁剪的图像
        # if save_crops:
        #     file_name = f'crop_{idx}.jpg'
        #     cropped_img = save_one_box(torch.tensor([x1, y1, x2, y2]), np.array(pil_image), file=Path(file_name))

    return frame_copy


# 加载预训练的YOLOv5模型
model_path = r'E:\8.8\yolov5-master\runs\train\exp15\weights\best.pt'
model = load_model(model_path)

# 指定摄像头设备，一般内置摄像头的索引是0
camera_index = 0

# 读取摄像头
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("无法打开摄像头")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break

        # 进行物体检测并绘制边界框
        annotated_frame = draw_boxes_and_calculate_dimensions(frame, model)

        # 显示处理后的帧
        cv2.imshow('Camera Stream', annotated_frame)

        # 按 Q 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 释放资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()