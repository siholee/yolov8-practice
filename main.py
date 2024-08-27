from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# YOLOv8 모델 로드
model = YOLO('models/yolov8n.pt')

# 이미지 파일 로드
image_path = '/Users/yurae/Documents/Coding/yolov8/images/road.webp'  # 이미지 파일의 절대 경로로 수정
image = cv2.imread(image_path)

# 이미지 로드 확인
if image is None:
    print(f"Error: Could not load image at {image_path}")
else:
    # 객체 검출
    results = model(image)

    # 검출된 객체 정보 출력
    for result in results:
        boxes = result.boxes.cpu().numpy()  
        for box in boxes:
            class_id = int(box.cls[0])  
            confidence = box.conf[0]  
            bbox = box.xyxy[0]  
            print(f'Class: {model.names[class_id]}, Confidence: {confidence:.2f}, BBox: {bbox}')

    # 결과 이미지 시각화
    result_image = results[0].plot()

    # Matplotlib으로 변환 후 출력
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
