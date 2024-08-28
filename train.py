import os
import yaml
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# YAML 파일 생성 함수
def create_yaml_file(dataset_path, output_path, class_names):
    data_yaml = {
        'train': os.path.join(dataset_path, 'images/train'),
        'val': os.path.join(dataset_path, 'images/val'),
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(output_path, 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=False)
    print(f"YAML file created at {output_path}")

# 모델 학습 함수
def train_model(data_yaml, epochs=100, img_size=640):
    # YOLOv8 모델 로드 및 학습
    model = YOLO('yolov8n.pt')  # 사전 학습된 모델을 로드
    results = model.train(data=data_yaml, epochs=epochs, imgsz=img_size)
    print("Training completed.")
    return model  # 학습된 모델 반환

# 이미지 추론 및 시각화 함수
def infer_and_visualize(model, image_path):
    # 이미지 파일 로드
    image = cv2.imread(image_path)
    
    # 이미지 로드 확인
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
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

    # BGR을 RGB로 변환하여 Matplotlib에 표시
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# 메인 함수
if __name__ == "__main__":
    # 데이터셋 경로 및 파라미터 설정
    dataset_path = os.path.abspath('dataset')  # 데이터셋 폴더의 절대 경로
    yaml_output_path = os.path.abspath('yaml/dataset.yaml')  # 생성할 YAML 파일 경로
    class_names = ['Box']  # 클래스 이름 리스트 (여기서는 'Box' 하나만 있음)
    
    # YAML 파일 생성
    create_yaml_file(dataset_path, yaml_output_path, class_names)
    
    # 모델 학습
    epochs = 100  # 학습 에폭 수
    img_size = 640  # 이미지 크기
    model = train_model(yaml_output_path, epochs, img_size)
    
    # 학습된 모델을 사용하여 이미지 추론 및 시각화
    image_path = os.path.abspath('images/logistics.jpg')  # 이미지 파일의 절대 경로로 수정
    infer_and_visualize(model, image_path)
