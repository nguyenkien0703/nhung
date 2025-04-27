from flask import Flask, request, render_template, session, url_for, redirect, jsonify
import os
from model import PlantDiseaseModel
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import io
import traceback
import shutil
import serial # Thêm thư viện serial
import threading # Thêm thư viện threading
import time # Thêm thư viện time
import json # Thêm thư viện json
from queue import Queue # Dùng Queue để trao đổi dữ liệu giữa các thread an toàn hơn

app = Flask(__name__)
app.secret_key = 'your-secret-key-123'

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Cấu hình Serial ---
SERIAL_PORT = 'COM5'  # *** Đã thay đổi thành COM5 ***
BAUD_RATE = 9600
# -----------------------

# Biến toàn cục để lưu trữ dữ liệu cảm biến mới nhất và trạng thái kết nối
latest_sensor_data = Queue(maxsize=1) # Sử dụng Queue để đảm bảo thread-safe và chỉ giữ giá trị mới nhất
arduino_connected = False
serial_thread = None
stop_thread = threading.Event() # Dùng Event để dừng thread an toàn

def read_arduino_data():
    """Hàm chạy trong thread nền để đọc dữ liệu từ Arduino."""
    global latest_sensor_data, arduino_connected
    ser = None
    while not stop_thread.is_set():
        if not arduino_connected:
            try:
                print(f"Dang thu ket noi toi Arduino tren {SERIAL_PORT}...")
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
                # Chờ Arduino khởi động lại sau khi kết nối Serial được thiết lập
                time.sleep(2)
                ser.reset_input_buffer() # Xóa bộ đệm đầu vào
                arduino_connected = True
                print(f"Da ket noi toi Arduino tren {SERIAL_PORT}.")
            except serial.SerialException as e:
                print(f"Loi ket noi Serial: {e}")
                arduino_connected = False
                ser = None # Đặt lại ser
                time.sleep(5) # Đợi 5 giây trước khi thử lại
                continue # Bỏ qua vòng lặp hiện tại và thử lại

        if ser and ser.is_open:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').rstrip()
                    # print(f"Nhan duoc: {line}") # Bỏ comment để debug
                    if line: # Đảm bảo dòng không rỗng
                        try:
                            data = json.loads(line)
                            # Kiểm tra xem có đủ các key cần thiết không
                            if all(k in data for k in ["temperature", "humidity", "light", "soil_moisture"]):
                                # Xóa item cũ nếu queue đầy và thêm item mới
                                if latest_sensor_data.full():
                                    latest_sensor_data.get_nowait() # Lấy ra item cũ
                                latest_sensor_data.put_nowait(data) # Đặt item mới
                                # print(f"Da cap nhat du lieu cam bien: {data}") # Bỏ comment để debug
                            elif "error" in data:
                                print(f"Loi tu Arduino: {data['error']}")
                            else:
                                print(f"Du lieu JSON khong hoan chinh: {line}")
                        except json.JSONDecodeError:
                            print(f"Loi giai ma JSON: {line}")
                        except Exception as e:
                            print(f"Loi xu ly du lieu: {e}")
                else:
                     # Nếu không có dữ liệu, đợi một chút
                     time.sleep(0.1)

            except serial.SerialException as e:
                print(f"Loi Serial trong khi doc: {e}")
                arduino_connected = False
                if ser:
                    ser.close()
                ser = None
                time.sleep(5) # Chờ trước khi thử kết nối lại
            except Exception as e:
                 print(f"Loi khong mong doi trong thread: {e}")
                 time.sleep(1) # Đợi chút để tránh vòng lặp lỗi nhanh

        else:
             # Nếu ser là None hoặc không mở, đợi trước khi thử kết nối lại
             time.sleep(5)

    # Dọn dẹp khi thread dừng
    if ser and ser.is_open:
        ser.close()
    print("Da dung thread doc Arduino.")


# Khởi tạo model
print("Khoi tao model...")
model = PlantDiseaseModel()
print("Model da san sang.")

# Khởi động thread đọc Serial
print("Khoi dong thread doc du lieu Arduino...")
serial_thread = threading.Thread(target=read_arduino_data, daemon=True) # daemon=True để thread tự thoát khi main thread thoát
serial_thread.start()
print("Thread doc du lieu Arduino da chay.")


@app.route('/api/sensor_data')
def get_sensor_data_api():
    """API endpoint để cung cấp dữ liệu cảm biến mới nhất và trạng thái kết nối."""
    global latest_sensor_data, arduino_connected
    data = None
    if not latest_sensor_data.empty():
        data = latest_sensor_data.queue[0]

    return jsonify({
        'connected': arduino_connected,
        'data': data
    })

@app.route('/')
def index():
    # Trang chủ hiển thị lựa chọn
    return render_template('index.html')

@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnosis():
    global latest_sensor_data, arduino_connected
    current_sensor_data = None
    arduino_status = "Đang kiểm tra..."

    # Lấy dữ liệu mới nhất từ queue (không chặn)
    if not latest_sensor_data.empty():
        current_sensor_data = latest_sensor_data.queue[0] # Lấy item mà không xóa
        arduino_status = "Đã nhận dữ liệu"
    elif not arduino_connected:
         arduino_status = "Chưa kết nối Arduino"

    # Chuyển saved_values sang lưu trữ các giá trị form nếu có
    saved_values = {
        'last_image': session.get('last_image', None),
        'current_image_path': session.get('current_image_path', None),
        'temperature': session.get('form_temperature', ''),
        'humidity': session.get('form_humidity', ''),
        'soil_moisture': session.get('form_soil_moisture', ''),
        'light': session.get('form_light', '')
    }

    prediction_result = None
    env_data_source = None
    temperature, humidity, soil_moisture, light = None, None, None, None

    if request.method == 'POST':
        # Xóa dữ liệu form cũ khỏi session trước khi xử lý POST mới
        session.pop('form_temperature', None)
        session.pop('form_humidity', None)
        session.pop('form_soil_moisture', None)
        session.pop('form_light', None)

        # Xác định nút nào được nhấn
        button_pressed = request.form.get('diagnose_button')

        # Xử lý dựa trên nút được nhấn
        if button_pressed == 'arduino':
            print("Nhan nut Chan doan (Arduino)")
            arduino_data = None
            if not latest_sensor_data.empty():
                arduino_data = latest_sensor_data.queue[0]
                if not all(k in arduino_data and arduino_data[k] is not None for k in ["temperature", "humidity", "light", "soil_moisture"]):
                    arduino_data = None
            
            if arduino_data:
                print("Su dung du lieu tu Arduino")
                temperature = arduino_data['temperature']
                humidity = arduino_data['humidity']
                soil_moisture = arduino_data['soil_moisture']
                light = arduino_data['light']
                env_data_source = 'arduino'
            else:
                # Lỗi: Không có dữ liệu Arduino hợp lệ khi người dùng yêu cầu
                arduino_status = "Chưa kết nối Arduino" if not arduino_connected else "Đang chờ dữ liệu hoặc dữ liệu không hợp lệ..."
                return render_template(
                    'diagnosis.html',
                    error="Không thể chẩn đoán bằng Arduino. Vui lòng kiểm tra kết nối hoặc dữ liệu cảm biến.",
                    saved_values=saved_values,
                    arduino_status=arduino_status,
                    sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
                )

        elif button_pressed == 'manual':
            print("Nhan nut Chan doan (Nhap tay)")
            # Lấy dữ liệu từ form
            temp_str = request.form.get('temperature')
            hum_str = request.form.get('humidity')
            soil_str = request.form.get('soil_moisture')
            light_str = request.form.get('light')

            # Lưu giá trị form vào session để hiển thị lại nếu có lỗi
            session['form_temperature'] = temp_str
            session['form_humidity'] = hum_str
            session['form_soil_moisture'] = soil_str
            session['form_light'] = light_str
            saved_values.update({ 'temperature': temp_str, 'humidity': hum_str, 'soil_moisture': soil_str, 'light': light_str })

            # Kiểm tra nhập đủ
            if not all([temp_str, hum_str, soil_str, light_str]):
                 arduino_status = "Chưa kết nối Arduino" if not arduino_connected else "Đang chờ dữ liệu..."
                 return render_template(
                    'diagnosis.html',
                    error="Vui lòng nhập đầy đủ thông số môi trường để chẩn đoán thủ công.",
                    saved_values=saved_values,
                    arduino_status=arduino_status,
                    sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
                 )
            # Kiểm tra kiểu dữ liệu
            try:
                temperature = float(temp_str)
                humidity = float(hum_str)
                soil_moisture = float(soil_str)
                light = float(light_str)
                env_data_source = 'manual'
                print(f"Su dung du lieu tu form: T={temperature}, H={humidity}, S={soil_moisture}, L={light}")
            except (ValueError, TypeError):
                 arduino_status = "Chưa kết nối Arduino" if not arduino_connected else "Đang chờ dữ liệu..."
                 return render_template(
                     'diagnosis.html',
                     error="Giá trị nhập vào cho thông số môi trường không hợp lệ.",
                     saved_values=saved_values,
                     arduino_status=arduino_status,
                     sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
                 )
        else:
            # Trường hợp không xác định được nút nào được nhấn (lỗi logic hoặc form?) 
            return render_template('diagnosis.html', error="Hành động không xác định.", saved_values=saved_values)

        # --- Tiếp tục xử lý ảnh và dự đoán ---
        try:
            if 'image' in request.files:
                image_file = request.files['image']
                if image_file and image_file.filename != '':
                    # Lưu ảnh
                    filename = secure_filename(image_file.filename)
                    image_path = os.path.join(UPLOAD_FOLDER, filename)
                    image_file.save(image_path)
                    
                    # Lưu đường dẫn ảnh vào session
                    session['current_image_path'] = image_path
                    session['last_image'] = filename
                    
                    # Đọc ảnh
                    img = Image.open(image_path)
                    
                    # Dự đoán
                    env_data = (temperature, humidity, soil_moisture, light)
                    prediction_result = model.predict(img, env_data)
                    
                    if prediction_result:
                        # Lưu kết quả dự đoán vào session
                        session['prediction_result'] = prediction_result
                        
                        # Chuẩn bị dữ liệu để hiển thị
                        filename = os.path.basename(image_path)
                        image_url = url_for('static', filename=f'uploads/{filename}')
                        
                        display_data = {
                            'plant_type': prediction_result['plant_type'],
                            'plant_confidence': f"{prediction_result['plant_confidence']*100:.2f}",
                            'disease': prediction_result['disease'],
                            'disease_confidence': f"{prediction_result['disease_confidence']*100:.2f}",
                            'image_disease': prediction_result['image_disease'],
                            'image_confidence': f"{prediction_result['image_confidence']*100:.2f}",
                            'env_disease': prediction_result['env_disease'],
                            'env_confidence': f"{prediction_result['env_confidence']*100:.2f}",
                            'image_url': image_url
                        }
                        
                        # Hiển thị kết quả trực tiếp trong trang diagnosis.html thay vì chuyển sang confirm.html
                        return render_template(
                            'diagnosis.html',
                            prediction=display_data,
                            saved_values=saved_values,
                            arduino_status=arduino_status,
                            sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
                        )
                    else:
                        return render_template(
                            'diagnosis.html',
                            error="Không thể thực hiện dự đoán. Vui lòng thử lại.",
                            saved_values=saved_values,
                            arduino_status=arduino_status,
                            sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
                        )
                else:
                    return render_template(
                        'diagnosis.html',
                        error="Vui lòng chọn ảnh để chẩn đoán.",
                        saved_values=saved_values,
                        arduino_status=arduino_status,
                        sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
                    )
            else:
                return render_template(
                    'diagnosis.html',
                    error="Không tìm thấy file ảnh trong yêu cầu.",
                    saved_values=saved_values,
                    arduino_status=arduino_status,
                    sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
                )
                
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh và dự đoán: {str(e)}")
            traceback.print_exc()
            return render_template(
                'diagnosis.html',
                error=f"Lỗi khi xử lý ảnh và dự đoán: {str(e)}",
                saved_values=saved_values,
                arduino_status=arduino_status,
                sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
            )

    # Nếu là GET request hoặc có lỗi, hiển thị trang chẩn đoán
    return render_template(
        'diagnosis.html',
        saved_values=saved_values,
        arduino_status=arduino_status,
        sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
    )


@app.route('/training', methods=['GET', 'POST'])
def training():
    global latest_sensor_data, arduino_connected
    current_sensor_data = None
    arduino_status = "Đang kiểm tra..."

    # Lấy dữ liệu mới nhất từ queue (không chặn)
    if not latest_sensor_data.empty():
        current_sensor_data = latest_sensor_data.queue[0]
        arduino_status = "Đã nhận dữ liệu"
    elif not arduino_connected:
         arduino_status = "Chưa kết nối Arduino"

    # Khởi tạo saved_values chỉ với ảnh (nếu có)
    saved_values = {
        'last_image': session.get('last_image', None),
        'current_image_path': session.get('current_image_path', None),
        'temperature': session.get('form_temperature', ''),
        'humidity': session.get('form_humidity', ''),
        'soil_moisture': session.get('form_soil_moisture', ''),
        'light': session.get('form_light', '')
    }

    prediction_result = None
    env_data_source = None
    session.pop('last_sensor_data', None) # Xóa dữ liệu cảm biến cũ trước khi xử lý mới
    session.pop('last_manual_data', None)

    if request.method == 'POST':
        # Xử lý hành động xác nhận huấn luyện
        if request.form.get('action') == 'confirm_training':
            try:
                # Lấy dữ liệu từ form
                image_path = request.form.get('image_path')
                
                # Chuyển đổi URL sang đường dẫn file thực
                if image_path:
                    if image_path.startswith('/static/uploads/'):
                        # Đây là URL tương đối, chuyển thành đường dẫn file
                        filename = image_path.split('/')[-1]
                        image_path = os.path.join(UPLOAD_FOLDER, filename)
                    elif image_path.startswith('static/uploads/'):
                        # Biến thể khác của URL
                        filename = image_path.split('/')[-1]
                        image_path = os.path.join(UPLOAD_FOLDER, filename)
                
                if not image_path or not os.path.exists(image_path):
                    # In ra để debug
                    print(f"ERROR: Không tìm thấy hình ảnh tại đường dẫn: {image_path}")
                    print(f"Form image_path đã gửi: {request.form.get('image_path')}")
                    
                    if 'current_image_path' in session:
                        # Thử sử dụng đường dẫn lưu trong session
                        image_path = session['current_image_path']
                        print(f"Thử sử dụng đường dẫn từ session: {image_path}")
                        
                        if not os.path.exists(image_path):
                            print(f"Đường dẫn từ session cũng không tồn tại.")
                    
                    return render_template(
                        'training.html',
                        error="Không tìm thấy hình ảnh đã tải lên. Vui lòng thử lại.",
                        saved_values=saved_values,
                        disease_classes=model.disease_classes,
                        plant_classes=model.plant_classes,
                        arduino_status=arduino_status
                    )
                
                # Lấy thông số môi trường
                temperature = float(request.form.get('temperature', 0))
                humidity = float(request.form.get('humidity', 0))
                soil_moisture = float(request.form.get('soil_moisture', 0))
                light = float(request.form.get('light', 0))
                
                # Lấy nhãn đúng
                correct_plant_idx = int(request.form.get('correct_plant', 0))
                disease_index = request.form.get('disease_index')
                
                # Mở ảnh
                img = Image.open(image_path)
                
                if disease_index == 'other':
                    # Xử lý trường hợp bệnh khác
                    other_disease = request.form.get('other_disease', '').strip()
                    if not other_disease:
                        return render_template(
                            'training.html',
                            error="Vui lòng nhập tên bệnh khi chọn 'Bệnh khác'.",
                            saved_values=saved_values,
                            disease_classes=model.disease_classes,
                            plant_classes=model.plant_classes,
                            arduino_status=arduino_status
                        )
                    
                    # Thêm bệnh mới vào danh sách
                    if other_disease not in model.disease_classes:
                        model.disease_classes.append(other_disease)
                    
                    # Lấy index của bệnh mới hoặc đã có
                    correct_disease_idx = model.disease_classes.index(other_disease)
                else:
                    correct_disease_idx = int(disease_index)
                
                # Thêm dữ liệu vào tập huấn luyện
                env_data = (temperature, humidity, soil_moisture, light)
                model.add_training_data(img, image_path, env_data, correct_disease_idx, correct_plant_idx)
                
                # Hiển thị thông báo thành công
                return render_template(
                    'training.html',
                    message=f"Đã thêm dữ liệu huấn luyện thành công! Bệnh: {model.disease_classes[correct_disease_idx]}, Loại cây: {model.plant_classes[correct_plant_idx]}",
                    saved_values=saved_values,
                    disease_classes=model.disease_classes,
                    plant_classes=model.plant_classes,
                    arduino_status=arduino_status
                )
                
            except Exception as e:
                error_info = traceback.format_exc()
                return render_template(
                    'training.html',
                    error=f"Lỗi khi xác nhận huấn luyện: {str(e)}",
                    details=error_info,
                    saved_values=saved_values,
                    disease_classes=model.disease_classes,
                    plant_classes=model.plant_classes,
                    arduino_status=arduino_status
                )
                
        # Xử lý nút chẩn đoán (không thay đổi)
        session.pop('form_temperature', None)
        session.pop('form_humidity', None)
        session.pop('form_soil_moisture', None)
        session.pop('form_light', None)

        button_pressed = request.form.get('diagnose_button')
        env_data_for_predict = None # Dữ liệu cuối cùng sẽ dùng để predict (dict)

        if button_pressed == 'arduino':
            print("Training - Nhan nut Chan doan (Arduino)")
            arduino_data = None
            if not latest_sensor_data.empty():
                arduino_data = latest_sensor_data.queue[0]
                if not all(k in arduino_data and arduino_data[k] is not None for k in ["temperature", "humidity", "light", "soil_moisture"]):
                    arduino_data = None
            
            if arduino_data:
                print("Training - Su dung du lieu tu Arduino")
                env_data_for_predict = arduino_data
                session['last_sensor_data'] = arduino_data
                env_data_source = 'arduino'
            else:
                arduino_status = "Chưa kết nối Arduino" if not arduino_connected else "Đang chờ dữ liệu hoặc dữ liệu không hợp lệ..."
                return render_template(
                    'training.html',
                    error="Không thể chẩn đoán bằng Arduino. Vui lòng kiểm tra kết nối hoặc dữ liệu cảm biến.",
                    saved_values=saved_values,
                    disease_classes=model.disease_classes,
                    plant_classes=model.plant_classes,
                    arduino_status=arduino_status,
                    sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
                )

        elif button_pressed == 'manual':
            print("Training - Nhan nut Chan doan (Nhap tay)")
            temp_str = request.form.get('temperature')
            hum_str = request.form.get('humidity')
            soil_str = request.form.get('soil_moisture')
            light_str = request.form.get('light')

            session['form_temperature'] = temp_str
            session['form_humidity'] = hum_str
            session['form_soil_moisture'] = soil_str
            session['form_light'] = light_str
            saved_values.update({ 'temperature': temp_str, 'humidity': hum_str, 'soil_moisture': soil_str, 'light': light_str })

            if not all([temp_str, hum_str, soil_str, light_str]):
                 arduino_status = "Chưa kết nối Arduino" if not arduino_connected else "Đang chờ dữ liệu..."
                 return render_template(
                     'training.html',
                     error="Vui lòng nhập đầy đủ thông số môi trường để chẩn đoán thủ công.",
                     saved_values=saved_values,
                     disease_classes=model.disease_classes,
                     plant_classes=model.plant_classes,
                     arduino_status=arduino_status,
                     sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
                 )
            try:
                manual_data_input = {
                    'temperature': float(temp_str),
                    'humidity': float(hum_str),
                    'soil_moisture': float(soil_str),
                    'light': float(light_str)
                }
                env_data_for_predict = manual_data_input
                session['last_manual_data'] = manual_data_input
                env_data_source = 'manual'
                print(f"Training - Su dung du lieu tu form: {manual_data_input}")
            except (ValueError, TypeError):
                 arduino_status = "Chưa kết nối Arduino" if not arduino_connected else "Đang chờ dữ liệu..."
                 return render_template(
                     'training.html',
                     error="Giá trị nhập vào cho thông số môi trường không hợp lệ.",
                     saved_values=saved_values,
                     disease_classes=model.disease_classes,
                     plant_classes=model.plant_classes,
                     arduino_status=arduino_status,
                     sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
                 )
        else:
             return render_template('training.html', 
                                  error="Hành động không xác định.", 
                                  saved_values=saved_values, 
                                  disease_classes=model.disease_classes,
                                  plant_classes=model.plant_classes)

        # --- Tiếp tục xử lý ảnh và dự đoán --- (Dùng env_data_for_predict)
        try:
            if 'image' in request.files:
                image_file = request.files['image']
                if image_file and image_file.filename != '':
                    # ... (lưu ảnh, xử lý ảnh)
                    filename = secure_filename(image_file.filename)
                    image_path = os.path.join(UPLOAD_FOLDER, filename)
                    image_file.save(image_path)
                    session['last_image'] = filename
                    session['current_image_path'] = image_path
                    saved_values['last_image'] = filename
                    saved_values['current_image_path'] = image_path

                    img = Image.open(image_path)
                    
                    # Tạo env_data từ dict env_data_for_predict
                    env_data = (env_data_for_predict['temperature'], env_data_for_predict['humidity'],
                               env_data_for_predict['soil_moisture'], env_data_for_predict['light'])

                    disease_pred = model.predict(img, env_data)
                    
                    # Xử lý kết quả trả về từ hàm predict
                    if disease_pred:
                        # Kiểm tra nếu có thông báo lỗi từ mô hình
                        if 'error' in disease_pred:
                            prediction_result = {
                                'error': disease_pred['message'] if 'message' in disease_pred else 'Không thể dự đoán với dữ liệu hiện tại',
                                'show_training_form': True,  # Vẫn hiển thị form huấn luyện ngay cả khi có lỗi
                                'image_path': url_for('static', filename=f'uploads/{filename}'),
                                'sensor_data': env_data_for_predict
                            }
                        else:
                            prediction_result = {
                                'disease': disease_pred['disease'],
                                'confidence': f"{disease_pred['disease_confidence']*100:.2f}",
                                'image_path': url_for('static', filename=f'uploads/{filename}'),
                                'sensor_data': env_data_for_predict,
                                'data_source': env_data_source,
                                'plant_type': disease_pred['plant_type'],
                                'plant_confidence': f"{disease_pred['plant_confidence']*100:.2f}",
                                'image_disease': disease_pred['image_disease'],
                                'image_confidence': f"{disease_pred['image_confidence']*100:.2f}",
                                'env_disease': disease_pred['env_disease'],
                                'env_confidence': f"{disease_pred['env_confidence']*100:.2f}",
                                'image_url': url_for('static', filename=f'uploads/{filename}')
                            }
                    else:
                        prediction_result = {
                            'error': 'Không thể thực hiện dự đoán. Vui lòng thử lại.',
                            'show_training_form': True,  # Vẫn hiển thị form huấn luyện khi có lỗi khác
                            'image_path': url_for('static', filename=f'uploads/{filename}'),
                            'sensor_data': env_data_for_predict
                        }
                else:
                    prediction_result = {'error': 'Vui lòng chọn một file hình ảnh hợp lệ.'}
            else:
                prediction_result = {'error': 'Không tìm thấy file hình ảnh trong yêu cầu.'}
        except Exception as e:
             error_info = traceback.format_exc()
             prediction_result = {
                 'error': f'Có lỗi xảy ra trong quá trình chẩn đoán: {str(e)}',
                 'details': error_info
             }
        # --- Kết thúc xử lý ảnh và dự đoán ---

    # --- Logic GET Request và Render Template (Giữ nguyên) ---
    # Lấy lại trạng thái arduino và data display cuối cùng trước khi render
    current_sensor_data_display = None
    if not latest_sensor_data.empty():
        current_sensor_data_display = latest_sensor_data.queue[0]
        arduino_status = "Đã nhận dữ liệu"
    elif not arduino_connected:
        arduino_status = "Chưa kết nối Arduino"
    else:
        arduino_status = "Đang chờ dữ liệu..."
    
    # Cập nhật saved_values lần cuối trước khi render (cho trường hợp GET hoặc POST bị lỗi trước khi predict)
    saved_values['last_image'] = session.get('last_image', None)
    saved_values['current_image_path'] = session.get('current_image_path', None)
    saved_values['temperature'] = session.get('form_temperature', '')
    saved_values['humidity'] = session.get('form_humidity', '')
    saved_values['soil_moisture'] = session.get('form_soil_moisture', '')
    saved_values['light'] = session.get('form_light', '')

    return render_template(
        'training.html',
        prediction=prediction_result,
        saved_values=saved_values,
        disease_classes=model.disease_classes,
        plant_classes=model.plant_classes,
        arduino_status=arduino_status,
        sensor_data_display=current_sensor_data_display
    )

@app.route('/confirm', methods=['GET', 'POST'])
def confirm_prediction():
    if request.method == 'POST':
        # Xử lý dữ liệu POST
        prediction_data = request.json
        if prediction_data:
            # Lưu dữ liệu vào session
            session['prediction_result'] = prediction_data
            return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'No data received'}), 400
    else:
        # Xử lý GET request
        prediction_result = session.get('prediction_result')
        if prediction_result:
            # Chuẩn bị dữ liệu để hiển thị
            if 'current_image_path' in session:
                image_path = session['current_image_path']
                filename = os.path.basename(image_path)
                image_url = url_for('static', filename=f'uploads/{filename}')
                
                display_data = {
                    'plant_type': prediction_result['plant_type'],
                    'plant_confidence': f"{prediction_result['plant_confidence']*100:.2f}%",
                    'disease': prediction_result['disease'],
                    'disease_confidence': f"{prediction_result['disease_confidence']*100:.2f}%",
                    'image_disease': prediction_result['image_disease'],
                    'image_confidence': f"{prediction_result['image_confidence']*100:.2f}%",
                    'env_disease': prediction_result['env_disease'],
                    'env_confidence': f"{prediction_result['env_confidence']*100:.2f}%",
                    'image_url': image_url
                }
                
                return render_template('confirm.html', prediction=display_data)
            else:
                return redirect(url_for('diagnosis'))
        return redirect(url_for('diagnosis'))

# Hàm để dừng thread an toàn khi tắt ứng dụng (ví dụ: Ctrl+C)
def signal_handler(sig, frame):
    print('Nhan duoc tin hieu dung, dang dung thread...')
    stop_thread.set()
    # Đợi thread kết thúc (có thể thêm timeout)
    if serial_thread and serial_thread.is_alive():
        serial_thread.join(timeout=5)
    print("Thread da dung. Thoat ung dung.")
    # Thoát chương trình
    os._exit(0) # Thoát ngay lập tức

import signal
signal.signal(signal.SIGINT, signal_handler) # Bắt Ctrl+C
signal.signal(signal.SIGTERM, signal_handler) # Bắt tín hiệu завершения

if __name__ == '__main__':
    # Không nên chạy debug=True khi có thread, có thể gây lỗi reload
    # app.run(debug=True)
    app.run(debug=False, host='0.0.0.0', port=5000) # Chạy trên tất cả các interface để có thể truy cập từ máy khác trong mạng

# ... (Phần confirm và các hàm khác giữ nguyên) ...

# ... (Phần confirm và các hàm khác giữ nguyên) ... 