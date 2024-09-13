from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog

# 이미지 파일 업로드 함수
def upload_image():
    # Tkinter로 파일 탐색기 창 열기
    Tk().withdraw()  # 기본 Tkinter 창 숨기기
    image_path = filedialog.askopenfilename(
        title="Select Image for Object Detection", 
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp")]  # 확장자 필터 수정
    )
    return image_path

# YOLOv8 모델 로드 (학습된 모델 logishub.pt 사용)
model = YOLO('models/logishub.pt')

# 이미지 파일 업로드
image_path = upload_image()

# 이미지 파일 로드 및 추론
if image_path:  # 이미지가 선택되었는지 확인
    image = cv2.imread(image_path)

    # 이미지 로드 확인
    if image is None:
        print(f"Error: Could not load image at {image_path}")
    else:
        # 객체 검출
        results = model(image)

        # 검출된 객체 정보 출력 및 이미지 시각화 준비
        for result in results:
            boxes = result.boxes.cpu().numpy()  # GPU 사용 시 CPU로 변환
            for box in boxes:
                class_id = int(box.cls[0])  # 클래스 ID
                confidence = box.conf[0]    # 신뢰도
                bbox = box.xyxy[0]          # 바운딩 박스 [xmin, ymin, xmax, ymax]
                print(f'Class: {model.names[class_id]}, Confidence: {confidence:.2f}, BBox: {bbox}')

        # 결과 이미지 시각화
        result_image = results[0].plot()

        # 이미지 시각화 (Matplotlib 사용)
        plt.figure(figsize=(10, 10))  # 이미지 크기 설정
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))  # BGR -> RGB 변환
        plt.axis('off')  # 축 제거
        plt.show()  # 이미지 출력
else:
    print("No image selected.")
