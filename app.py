from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io

app = Flask(__name__)
CORS(app)

# YOLO 모델 로드
model = YOLO("models/logishub.pt")

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # 파일 읽기 및 YOLO 모델로 이미지 처리
        image = Image.open(io.BytesIO(file.read()))
        results = model(image)
        counts = {}

        # 이미지를 복사하여 그리기 객체 만들기
        draw = ImageDraw.Draw(image)

        # 결과를 순회하면서 bounding box와 카운트를 처리
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                counts[class_name] = counts.get(class_name, 0) + 1

                # Bounding box 좌표 가져오기
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표 변환
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1), f"{class_name} {box.conf[0]:.2f}", fill="red")

        # 결과 이미지 저장
        image_path = 'static/detected_image.jpg'
        image.save(image_path)

        # JSON 응답과 함께 이미지 파일 경로 제공
        return jsonify({
            'detections': counts,
            'image_url': '13.211.211.71/static/detected_image.jpg'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
