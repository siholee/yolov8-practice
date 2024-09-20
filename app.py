import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Flask 설정
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = 'supersecretkey'
model = YOLO('models/logishub.pt')  # YOLO 모델 로드

# 허용된 이미지 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# 확장자 확인 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 메인 페이지 라우트
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 파일이 제출되었는지 확인
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # 파일 저장
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 파일 경로 확인
            print(f"Uploaded file path: {filepath}")
            
            # 객체 검출 수행
            image = cv2.imread(filepath)
            
            if image is None:
                flash('Error: Unable to read image')
                return redirect(request.url)
            
            results = model(image)

            # 결과 시각화 이미지 생성
            result_image = results[0].plot()

            # BGR -> RGB 변환 (OpenCV와 Matplotlib 호환)
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # 저장할 파일 경로에 확장자 추가
            output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{filename}")
            
            # 파일 확장자를 명확하게 지정 (png 형식으로 저장)
            if not output_image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                output_image_path += '.png'  # 확장자가 없으면 .png 추가
            
            # 이미지 저장
            cv2.imwrite(output_image_path, result_image_rgb)

            # 추론된 박스 수 확인
            detected_boxes = len(results[0].boxes)

            # 결과를 클라이언트에 전달
            return render_template('index.html', uploaded_image=filename, output_image=os.path.basename(output_image_path), detected_boxes=detected_boxes)

    return render_template('index.html')

# 앱 실행
if __name__ == '__main__':
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')  # 업로드 폴더가 없으면 생성
    app.run(debug=True)
