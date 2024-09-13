import os
import yaml
from ultralytics import YOLO

# YAML 파일 생성 함수
def create_yaml_file(dataset_path, output_path, class_names):
    data_yaml = {
        'train': os.path.join(dataset_path, 'train/images'),
        'val': os.path.join(dataset_path, 'valid/images'),
        'test': os.path.join(dataset_path, 'test/images'),
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
    results = model.train(data=data_yaml, epochs=epochs, imgsz=img_size, name="logishub")
    print("Training completed.")
    
    # 학습된 모델 저장
    model_output_path = os.path.abspath('models/logishub.pt')  # 모델이 저장될 경로 설정
    model.save(model_output_path)  # 모델 저장
    print(f"Model saved at {model_output_path}")
    return model  # 학습된 모델 반환

# 메인 함수
if __name__ == "__main__":
    # 데이터셋 경로 및 파라미터 설정
    dataset_path = os.path.abspath('dataset')  # 데이터셋 폴더의 절대 경로
    yaml_output_path = os.path.abspath('yaml/dataset.yaml')  # 생성할 YAML 파일 경로
    class_names = ['box']  # 클래스 이름 리스트 (여기서는 'box' 하나만 있음)
    
    # YAML 파일 생성
    create_yaml_file(dataset_path, yaml_output_path, class_names)
    
    # 모델 학습
    epochs = 100  # 학습 에폭 수
    img_size = 640  # 이미지 크기
    train_model(yaml_output_path, epochs, img_size)
