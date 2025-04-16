from flask import Flask, request, render_template, session, url_for
import os
from model import PlantDiseaseModel
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import io
import traceback
import shutil

app = Flask(__name__)
app.secret_key = 'your-secret-key-123'  # Cần thiết cho session

# Tạo thư mục để lưu ảnh
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Khởi tạo model
model = PlantDiseaseModel()

@app.route('/', methods=['GET', 'POST'])
def home():
    # Khởi tạo giá trị mặc định cho saved_values
    if 'temperature' not in session:
        session['temperature'] = ''
    if 'humidity' not in session:
        session['humidity'] = ''
    if 'soil_moisture' not in session:
        session['soil_moisture'] = ''
    if 'light' not in session:
        session['light'] = ''
    if 'last_image' not in session:
        session['last_image'] = None
    if 'current_image_path' not in session:
        session['current_image_path'] = None

    saved_values = {
        'temperature': session.get('temperature', ''),
        'humidity': session.get('humidity', ''),
        'soil_moisture': session.get('soil_moisture', ''),
        'light': session.get('light', ''),
        'last_image': session.get('last_image', None),
        'current_image_path': session.get('current_image_path', None)
    }
    
    prediction_result = None
    
    if request.method == 'POST':
        try:
            # Lưu các giá trị vào session trước khi xử lý
            session['temperature'] = request.form.get('temperature', '')
            session['humidity'] = request.form.get('humidity', '')
            session['soil_moisture'] = request.form.get('soil_moisture', '')
            session['light'] = request.form.get('light', '')
            
            # Lấy dữ liệu từ form
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            soil_moisture = float(request.form['soil_moisture'])
            light = float(request.form['light'])
            
            # Xử lý file ảnh
            if 'image' in request.files:
                image_file = request.files['image']
                if image_file:
                    # Tạo tên file an toàn
                    filename = secure_filename(image_file.filename)
                    
                    # Tạo đường dẫn đầy đủ
                    image_path = os.path.join(UPLOAD_FOLDER, filename)
                    
                    # Lưu file
                    image_file.save(image_path)
                    
                    # Lưu thông tin vào session
                    session['last_image'] = filename
                    session['current_image_path'] = image_path
                    
                    # Đọc và xử lý ảnh
                    img = Image.open(image_path)
                    img = img.resize((224, 224))  # Resize ảnh về kích thước 224x224
                    img = np.array(img)
                    
                    # Tạo dữ liệu môi trường
                    env_data = np.array([temperature, humidity, soil_moisture, light])
                    
                    # Dự đoán
                    disease_pred = model.predict(img, env_data)
                    
                    # Lấy kết quả dự đoán
                    pred_disease = np.argmax(disease_pred[0])
                    confidence = disease_pred[0][pred_disease]
                    
                    prediction_result = {
                        'disease': model.disease_classes[pred_disease],
                        'confidence': f"{confidence*100:.2f}",
                        'image_path': url_for('static', filename=f'uploads/{filename}')
                    }
                    
                    # Thêm xác suất cho từng loại bệnh
                    prediction_result['probabilities'] = {
                        disease: f"{prob*100:.2f}%" 
                        for disease, prob in zip(model.disease_classes, disease_pred[0])
                    }
            else:
                prediction_result = {'error': 'Vui lòng chọn một hình ảnh'}
        
        except Exception as e:
            # Lấy thông tin chi tiết về lỗi
            error_info = traceback.format_exc()
            prediction_result = {
                'error': f'Có lỗi xảy ra: {str(e)}',
                'details': error_info
            }
    
    # Cập nhật saved_values với giá trị mới nhất từ session
    saved_values = {
        'temperature': session.get('temperature', ''),
        'humidity': session.get('humidity', ''),
        'soil_moisture': session.get('soil_moisture', ''),
        'light': session.get('light', ''),
        'last_image': session.get('last_image', None),
        'current_image_path': session.get('current_image_path', None)
    }
    
    return render_template(
        'index.html',
        prediction=prediction_result,
        saved_values=saved_values
    )

@app.route('/confirm', methods=['POST'])
def confirm_prediction():
    # Khởi tạo saved_values từ session
    saved_values = {
        'temperature': session.get('temperature', ''),
        'humidity': session.get('humidity', ''),
        'soil_moisture': session.get('soil_moisture', ''),
        'light': session.get('light', ''),
        'last_image': session.get('last_image', None),
        'current_image_path': session.get('current_image_path', None)
    }
    
    try:
        # Lấy thông tin từ form
        disease_index = request.form['disease_index']
        
        # Lấy thông tin từ session
        image_path = session.get('current_image_path')
        temperature = float(session.get('temperature'))
        humidity = float(session.get('humidity'))
        soil_moisture = float(session.get('soil_moisture'))
        light = float(session.get('light'))
        
        if image_path and os.path.exists(image_path):
            # Tạo dữ liệu môi trường
            env_data = [temperature, humidity, soil_moisture, light]
            
            # Xử lý trường hợp bệnh khác
            if disease_index == 'other':
                other_disease = request.form.get('other_disease', '').strip()
                if not other_disease:
                    return render_template(
                        'index.html',
                        error="Vui lòng nhập tên bệnh khác",
                        saved_values=saved_values
                    )
                
                # Thêm bệnh mới vào danh sách bệnh của model
                if other_disease not in model.disease_classes:
                    model.disease_classes.append(other_disease)
                    # Cập nhật lại mô hình với lớp mới
                    model.image_model = model._build_cnn_model()
                
                # Lấy index của bệnh mới
                disease_index = model.disease_classes.index(other_disease)
            else:
                disease_index = int(disease_index)
            
            # Thêm dữ liệu vào tập training
            model.add_training_data(
                image=open(image_path, 'rb').read(),
                image_path=image_path,
                env_data=env_data,
                label=disease_index
            )
            
            # Tạo prediction_result để hiển thị lại ảnh
            prediction_result = None
            if session.get('last_image'):
                prediction_result = {
                    'image_path': url_for('static', filename=f'uploads/{session["last_image"]}')
                }
            
            return render_template(
                'index.html',
                message="Đã thêm dữ liệu vào tập huấn luyện và cập nhật mô hình thành công!",
                saved_values=saved_values,
                prediction=prediction_result
            )
        else:
            return render_template(
                'index.html',
                error="Không tìm thấy ảnh đã upload",
                saved_values=saved_values
            )
            
    except Exception as e:
        return render_template(
            'index.html',
            error=f"Lỗi khi xác nhận kết quả: {str(e)}",
            saved_values=saved_values
        )

if __name__ == '__main__':
    app.run(debug=True) 