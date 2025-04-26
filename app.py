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

        # --- Tiếp tục xử lý ảnh và dự đoán --- (Logic này dùng chung temperature, humidity, etc. đã được xác định ở trên)
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
                    img = img.resize((224, 224))
                    img = np.array(img)

                    env_data = np.array([temperature, humidity, soil_moisture, light])

                    disease_pred = model.predict(img, env_data)
                    pred_disease = np.argmax(disease_pred[0])
                    confidence = disease_pred[0][pred_disease]

                    prediction_result = {
                        'disease': model.disease_classes[pred_disease],
                        'confidence': f"{confidence*100:.2f}",
                        'image_path': url_for('static', filename=f'uploads/{filename}'),
                        'sensor_data': {
                            'temperature': temperature,
                            'humidity': humidity,
                            'soil_moisture': soil_moisture,
                            'light': light
                        },
                        'data_source': env_data_source
                    }
                    prediction_result['probabilities'] = { disease: f"{prob*100:.2f}%" for disease, prob in zip(model.disease_classes, disease_pred[0]) }
                    
                    # Xóa dữ liệu form khỏi session nếu thành công (vì đã dùng hoặc lấy từ Arduino)
                    session.pop('form_temperature', None)
                    session.pop('form_humidity', None)
                    session.pop('form_soil_moisture', None)
                    session.pop('form_light', None)
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
        'diagnosis.html',
        prediction=prediction_result,
        saved_values=saved_values,
        arduino_status=arduino_status,
        sensor_data_display=current_sensor_data_display
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
                     arduino_status=arduino_status,
                     sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
                 )
        else:
             return render_template('training.html', error="Hành động không xác định.", saved_values=saved_values, disease_classes=model.disease_classes)

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
                    img = img.resize((224, 224))
                    img = np.array(img)

                    # Tạo env_data từ dict env_data_for_predict
                    env_data = np.array([env_data_for_predict['temperature'], env_data_for_predict['humidity'],
                                         env_data_for_predict['soil_moisture'], env_data_for_predict['light']])

                    disease_pred = model.predict(img, env_data)
                    pred_disease = np.argmax(disease_pred[0])
                    confidence = disease_pred[0][pred_disease]

                    prediction_result = {
                        'disease': model.disease_classes[pred_disease],
                        'confidence': f"{confidence*100:.2f}",
                        'image_path': url_for('static', filename=f'uploads/{filename}'),
                        'sensor_data': env_data_for_predict,
                        'data_source': env_data_source
                    }
                    prediction_result['probabilities'] = { disease: f"{prob*100:.2f}%" for disease, prob in zip(model.disease_classes, disease_pred[0]) }
                    
                    # Xóa dữ liệu form khỏi session nếu thành công
                    session.pop('form_temperature', None)
                    session.pop('form_humidity', None)
                    session.pop('form_soil_moisture', None)
                    session.pop('form_light', None)
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
        arduino_status=arduino_status,
        sensor_data_display=current_sensor_data_display
    )

@app.route('/confirm', methods=['POST'])
def confirm_prediction():
    saved_values = { # Chỉ cần lấy ảnh từ session
        'last_image': session.get('last_image', None),
        'current_image_path': session.get('current_image_path', None),
        # Thêm giá trị form để hiển thị lại nếu confirm lỗi
        'temperature': session.get('form_temperature', ''),
        'humidity': session.get('form_humidity', ''),
        'soil_moisture': session.get('form_soil_moisture', ''),
        'light': session.get('form_light', '')
    }
    # Xác định nguồn dữ liệu đã dùng và lấy dữ liệu đó
    env_data_to_train = None
    data_source = None
    last_sensor_data = session.get('last_sensor_data')
    last_manual_data = session.get('last_manual_data')

    if last_sensor_data:
        env_data_to_train = last_sensor_data
        data_source = 'arduino'
        print("Confirm - Su dung du lieu Arduino da luu")
    elif last_manual_data:
        env_data_to_train = last_manual_data
        data_source = 'manual'
        print(f"Confirm - Su dung du lieu manual da luu: {last_manual_data}")
    else:
        # Trường hợp không có dữ liệu nào được lưu (lỗi logic trước đó?)
        return render_template(
            'training.html',
            error="Không tìm thấy dữ liệu môi trường đã sử dụng cho lần chẩn đoán trước.",
            saved_values=saved_values,
            disease_classes=model.disease_classes,
            # Cần lấy lại trạng thái Arduino hiện tại để hiển thị
            arduino_status = "Đã nhận dữ liệu" if not latest_sensor_data.empty() else ("Chưa kết nối Arduino" if not arduino_connected else "Đang chờ dữ liệu..."),
            sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None
        )

    arduino_status_display = "Đã sử dụng dữ liệu Arduino" if data_source == 'arduino' else "Đã sử dụng dữ liệu nhập tay"

    try:
        disease_index_str = request.form['disease_index']
        image_path = session.get('current_image_path')

        # Lấy các giá trị từ env_data_to_train
        temperature = env_data_to_train.get('temperature')
        humidity = env_data_to_train.get('humidity')
        soil_moisture = env_data_to_train.get('soil_moisture')
        light = env_data_to_train.get('light')

        if image_path and os.path.exists(image_path) and None not in [temperature, humidity, soil_moisture, light]:
            env_data_list = [temperature, humidity, soil_moisture, light] # Dùng list cho add_training_data

            if disease_index_str == 'other':
                other_disease = request.form.get('other_disease', '').strip()
                if not other_disease:
                    # Prediction cũ để hiển thị lại
                    prediction_result = {
                         'image_path': url_for('static', filename=f'uploads/{session.get("last_image")}') if session.get("last_image") else None,
                         'sensor_data': env_data_to_train, # Dữ liệu đã dùng
                         'data_source': data_source
                    }
                    return render_template(
                        'training.html',
                        error="Vui lòng nhập tên bệnh khác",
                        saved_values=saved_values,
                        disease_classes=model.disease_classes,
                        prediction=prediction_result, # Gửi lại prediction cũ
                        arduino_status=arduino_status_display
                    )

                if other_disease not in model.disease_classes:
                    # ... (logic thêm bệnh mới, xây lại model)
                     model.disease_classes.append(other_disease)
                     print("Xay dung lai mo hinh CNN do co lop moi...")
                     model.image_model = model._build_cnn_model()
                     print("Luu y: Can huan luyen lai toan bo mo hinh voi du lieu day du.")
                     model.save_training_data()

                disease_index = model.disease_classes.index(other_disease)
            else:
                disease_index = int(disease_index_str)
                if disease_index < 0 or disease_index >= len(model.disease_classes):
                     raise ValueError("Chỉ số bệnh không hợp lệ.")

            with open(image_path, 'rb') as f:
                 image_bytes = f.read()

            print(f"Them du lieu training: image={os.path.basename(image_path)}, env={env_data_list}, label_idx={disease_index}, label={model.disease_classes[disease_index]}")

            model.add_training_data(
                image=image_bytes,
                image_path=image_path,
                env_data=env_data_list,
                label=disease_index
            )

            # Xóa dữ liệu đã dùng khỏi session
            if data_source == 'arduino':
                session.pop('last_sensor_data', None)
            elif data_source == 'manual':
                session.pop('last_manual_data', None)
            # Xóa luôn dữ liệu form nếu còn sót
            session.pop('form_temperature', None)
            session.pop('form_humidity', None)
            session.pop('form_soil_moisture', None)
            session.pop('form_light', None)

            prediction_result = None
            if session.get('last_image'):
                 prediction_result = {
                    'image_path': url_for('static', filename=f'uploads/{session.get("last_image")}')
                 }

            return render_template(
                'training.html',
                message="Đã thêm dữ liệu vào tập huấn luyện và cập nhật mô hình thành công!",
                saved_values=saved_values, # Chỉ còn ảnh
                prediction=prediction_result, # Chỉ hiển thị lại ảnh
                disease_classes=model.disease_classes,
                arduino_status="Nhập dữ liệu mới để huấn luyện tiếp", # Reset status
                sensor_data_display=latest_sensor_data.queue[0] if not latest_sensor_data.empty() else None # Hiển thị data mới nhất từ Arduino nếu có
            )
        else:
            error_msg = "Không tìm thấy ảnh đã upload hoặc dữ liệu môi trường không hợp lệ."
            # ... (logic tạo prediction_result cũ để hiển thị lại)
            prediction_result = {
                 'image_path': url_for('static', filename=f'uploads/{session.get("last_image")}') if session.get("last_image") else None,
                 'sensor_data': env_data_to_train,
                 'data_source': data_source
            }
            return render_template(
                'training.html',
                error=error_msg,
                saved_values=saved_values,
                disease_classes=model.disease_classes,
                prediction=prediction_result, # Gửi lại prediction cũ
                arduino_status=arduino_status_display
            )

    except KeyError as ke:
         # ... (xử lý lỗi KeyError, tương tự như cũ nhưng dùng arduino_status_display)
         error_msg = f"Thiếu dữ liệu trong form hoặc session: {str(ke)}"
         prediction_result = {
             'image_path': url_for('static', filename=f'uploads/{session.get("last_image")}') if session.get("last_image") else None,
             'sensor_data': env_data_to_train,
             'data_source': data_source
         }
         return render_template(
            'training.html',
            error=error_msg,
            saved_values=saved_values,
            disease_classes=model.disease_classes,
            prediction=prediction_result,
            arduino_status=arduino_status_display
         )
    except Exception as e:
        # ... (xử lý lỗi Exception, tương tự như cũ nhưng dùng arduino_status_display)
        error_info = traceback.format_exc()
        print(f"ERROR in /confirm: {error_info}")
        prediction_result = {
            'image_path': url_for('static', filename=f'uploads/{session.get("last_image")}') if session.get("last_image") else None,
            'sensor_data': env_data_to_train,
            'data_source': data_source
        }
        return render_template(
            'training.html',
            error=f"Lỗi khi xác nhận kết quả: {str(e)}",
            saved_values=saved_values,
            disease_classes=model.disease_classes,
            prediction=prediction_result,
            arduino_status=arduino_status_display
        )

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