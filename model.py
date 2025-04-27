import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
import cv2
from PIL import Image
import io
import os
import json
import pickle
import shutil
import traceback

class PlantDiseaseModel:
    def __init__(self):
        # Định nghĩa danh sách các loại cây
        self.plant_classes = [
            "Cây lúa",
            "Cây cà chua",
            "Cây ớt",
            "Cây khoai tây",
            "Cây dưa chuột",
            "Cây nho"
        ]
        
        # Định nghĩa danh sách các loại bệnh
        self.disease_classes = [
            "Khỏe mạnh",
            "Bệnh đốm lá",
            "Bệnh thối rễ",
            "Bệnh nấm mốc",
            "Bệnh virus",
            "Bệnh đạo ôn",
            "Bệnh bạc lá",
            "Bệnh khô vằn",
            "Bệnh mốc sương",
            "Bệnh thán thư",
            "Bệnh héo xanh",
            "Bệnh phấn trắng",
            "Bệnh sương mai",
            "Bệnh gỉ sắt",
            "Bệnh ghẻ"
        ]
        
        # Khởi tạo mô hình CNN cho xử lý hình ảnh
        self.image_model = None
        self.plant_type_model = None  # Thêm mô hình nhận diện loại cây
        
        # Khởi tạo mô hình Random Forest
        self.env_model = None
        
        # Khởi tạo danh sách dữ liệu training
        self.training_data = {
            'images': [],
            'env_data': [],
            'labels': [],
            'image_paths': [],
            'plant_types': []
        }
        
        # Tải dữ liệu training và mô hình nếu có
        if self.load_training_data():
            print("Đã tải mô hình và dữ liệu training thành công!")
        else:
            print("Khởi tạo mô hình mới...")
            # Khởi tạo mô hình mới
            self.image_model = self._build_cnn_model()
            self.plant_type_model = self._build_plant_type_model()  # Khởi tạo mô hình nhận diện loại cây
            self.env_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Lưu mô hình trống (không tạo dữ liệu mẫu ngẫu nhiên)
            print("Khởi tạo mô hình mới không có dữ liệu mẫu...")
            self.save_training_data()
    
    def _build_cnn_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(len(self.disease_classes), activation='softmax')
        ])
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def _build_plant_type_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(len(self.plant_classes), activation='softmax')
        ])
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def add_training_data(self, image, image_path, env_data, label, plant_type_idx=0):
        """Thêm dữ liệu mới vào tập training"""
        try:
            # Tạo thư mục data/images nếu chưa tồn tại
            images_dir = 'data/images'
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            
            # Tạo tên file mới cho ảnh trong thư mục data/images
            image_filename = os.path.basename(image_path)
            new_image_path = os.path.join(images_dir, image_filename)
            
            # Sao chép ảnh vào thư mục data/images
            shutil.copy2(image_path, new_image_path)
            
            # Thêm dữ liệu môi trường
            self.training_data['env_data'].append(env_data)
            
            # Thêm nhãn bệnh
            label_one_hot = np.zeros(len(self.disease_classes))
            label_one_hot[label] = 1
            self.training_data['labels'].append(label_one_hot.tolist())
            
            # Thêm đường dẫn ảnh mới
            self.training_data['image_paths'].append(new_image_path)
            
            # Nếu có thông tin về plant_type trong training_data
            if 'plant_types' not in self.training_data:
                self.training_data['plant_types'] = []
            
            # Thêm thông tin loại cây
            plant_type_one_hot = np.zeros(len(self.plant_classes))
            plant_type_one_hot[plant_type_idx] = 1
            self.training_data['plant_types'].append(plant_type_one_hot.tolist())
            
            # Lưu dữ liệu trước khi huấn luyện
            self.save_training_data()
            
            # Huấn luyện lại mô hình
            self.retrain()
            
        except Exception as e:
            raise Exception(f"Lỗi khi thêm dữ liệu training: {str(e)}")
    
    def retrain(self):
        """Huấn luyện lại mô hình với tất cả dữ liệu"""
        try:
            if len(self.training_data['labels']) > 0:
                # Tải lại tất cả ảnh
                images = []
                valid_indices = []
                
                for i, image_path in enumerate(self.training_data['image_paths']):
                    if os.path.exists(image_path):
                        try:
                            img = Image.open(image_path)
                            img = img.resize((224, 224))
                            img = np.array(img)
                            if img.dtype != np.float32 or img.max() > 1.0:
                                img = img.astype('float32') / 255.0
                            images.append(img)
                            valid_indices.append(i)
                        except Exception as e:
                            print(f"Lỗi khi đọc ảnh {image_path}: {str(e)}")
                            continue
                
                if not images:
                    raise Exception("Không có ảnh hợp lệ để huấn luyện")
                
                # Lọc dữ liệu chỉ giữ lại các mẫu có ảnh hợp lệ
                X_img = np.array(images)
                X_env = np.array([self.training_data['env_data'][i] for i in valid_indices])
                y_disease = np.array([self.training_data['labels'][i] for i in valid_indices])
                
                # Lấy dữ liệu loại cây nếu có
                y_plant = None
                if 'plant_types' in self.training_data and len(self.training_data['plant_types']) > 0:
                    y_plant = np.array([self.training_data['plant_types'][i] for i in valid_indices])
                
                # Cập nhật lại training_data
                self.training_data['env_data'] = [self.training_data['env_data'][i] for i in valid_indices]
                self.training_data['labels'] = [self.training_data['labels'][i] for i in valid_indices]
                self.training_data['image_paths'] = [self.training_data['image_paths'][i] for i in valid_indices]
                if 'plant_types' in self.training_data:
                    self.training_data['plant_types'] = [self.training_data['plant_types'][i] for i in valid_indices]
                
                # Huấn luyện mô hình
                self.train(X_img, X_env, y_plant, y_disease)
                
                # Lưu mô hình và dữ liệu sau khi huấn luyện
                self.save_training_data()
                
        except Exception as e:
            raise Exception(f"Lỗi khi huấn luyện lại mô hình: {str(e)}")
    
    def save_training_data(self):
        """Lưu dữ liệu training và mô hình vào file"""
        try:
            # Tạo thư mục data nếu chưa tồn tại
            if not os.path.exists('data'):
                os.makedirs('data')
            
            # Lưu dữ liệu môi trường và nhãn
            data = {
                'env_data': self.training_data['env_data'],
                'labels': self.training_data['labels'],
                'image_paths': self.training_data['image_paths'],
                'plant_types': self.training_data['plant_types'],
                'disease_classes': self.disease_classes,  # Lưu thêm danh sách bệnh
                'plant_classes': self.plant_classes      # Lưu thêm danh sách cây
            }
            
            with open('data/training_data.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Lưu mô hình CNN
            if self.image_model is not None:
                if not os.path.exists('data/image_model'):
                    os.makedirs('data/image_model')
                self.image_model.save('data/image_model')
            
            # Lưu mô hình nhận diện loại cây
            if self.plant_type_model is not None:
                if not os.path.exists('data/plant_type_model'):
                    os.makedirs('data/plant_type_model')
                self.plant_type_model.save('data/plant_type_model')
            
            # Lưu mô hình Random Forest
            if self.env_model is not None:
                with open('data/env_model.pkl', 'wb') as f:
                    pickle.dump(self.env_model, f)
                
            print("Đã lưu dữ liệu training và mô hình thành công!")
                
        except Exception as e:
            print(f"Lỗi khi lưu dữ liệu training: {str(e)}")
    
    def load_training_data(self):
        """Tải dữ liệu training và mô hình từ file. Trả về True nếu tải thành công."""
        try:
            data_loaded = False
            model_loaded = False
            
            # Tải dữ liệu training
            if os.path.exists('data/training_data.json'):
                with open('data/training_data.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Kiểm tra và lọc ra các ảnh còn tồn tại
                    valid_indices = []
                    for i, image_path in enumerate(data['image_paths']):
                        if os.path.exists(image_path):
                            valid_indices.append(i)
                    
                    # Chỉ tải dữ liệu của các ảnh còn tồn tại
                    self.training_data['env_data'] = [data['env_data'][i] for i in valid_indices]
                    self.training_data['labels'] = [data['labels'][i] for i in valid_indices]
                    self.training_data['image_paths'] = [data['image_paths'][i] for i in valid_indices]
                    self.training_data['plant_types'] = [data['plant_types'][i] for i in valid_indices]
                    
                    # Tải lại danh sách bệnh và cây
                    if 'disease_classes' in data:
                        self.disease_classes = data['disease_classes']
                    if 'plant_classes' in data:
                        self.plant_classes = data['plant_classes']
                        
                    print(f"Đã tải {len(valid_indices)} mẫu dữ liệu training")
                    data_loaded = len(valid_indices) > 0
            
            # Tải mô hình CNN nếu có
            if os.path.exists('data/image_model'):
                try:
                    print("Đang tải mô hình CNN...")
                    self.image_model = tf.keras.models.load_model('data/image_model')
                    model_loaded = True
                except Exception as e:
                    print(f"Lỗi khi tải mô hình CNN: {str(e)}")
                    self.image_model = self._build_cnn_model()
            else:
                print("Khởi tạo mô hình CNN mới...")
                self.image_model = self._build_cnn_model()
            
            # Tải mô hình nhận diện loại cây nếu có
            if os.path.exists('data/plant_type_model'):
                try:
                    print("Đang tải mô hình nhận diện loại cây...")
                    self.plant_type_model = tf.keras.models.load_model('data/plant_type_model')
                    model_loaded = model_loaded and True
                except Exception as e:
                    print(f"Lỗi khi tải mô hình nhận diện loại cây: {str(e)}")
                    self.plant_type_model = self._build_plant_type_model()
            else:
                print("Khởi tạo mô hình nhận diện loại cây mới...")
                self.plant_type_model = self._build_plant_type_model()
            
            # Tải mô hình Random Forest nếu có
            if os.path.exists('data/env_model.pkl'):
                try:
                    print("Đang tải mô hình Random Forest...")
                    with open('data/env_model.pkl', 'rb') as f:
                        self.env_model = pickle.load(f)
                    model_loaded = model_loaded and True
                except Exception as e:
                    print(f"Lỗi khi tải mô hình Random Forest: {str(e)}")
                    self.env_model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                print("Khởi tạo mô hình Random Forest mới...")
                self.env_model = RandomForestClassifier(n_estimators=100, random_state=42)
                
            return data_loaded and model_loaded
                
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu training: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """Tiền xử lý hình ảnh cho mô hình"""
        try:
            # Chuyển đổi PIL Image sang numpy array nếu cần
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Resize ảnh nếu cần
            if image.shape[:2] != (224, 224):
                image = cv2.resize(image, (224, 224))
            
            # Chuyển đổi sang float32 và chuẩn hóa
            if image.dtype != np.float32 or image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            
            # Thêm batch dimension nếu cần
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"Lỗi khi tiền xử lý ảnh: {str(e)}")
            traceback.print_exc()
            raise
    
    def preprocess_environment_data(self, temperature, humidity, soil_moisture, light):
        try:
            # Chuẩn hóa dữ liệu môi trường
            env_data = [
                float(temperature),
                float(humidity),
                float(soil_moisture),
                float(light)
            ]
            return env_data
        except Exception as e:
            raise Exception(f"Lỗi xử lý dữ liệu môi trường: {str(e)}")
    
    def train(self, images, env_data, plant_labels, disease_labels, epochs=10, batch_size=32):
        print("Training CNN models...")
        # Ensure images are properly preprocessed
        processed_images = []
        for img in images:
            processed_img = self.preprocess_image(img)
            processed_images.append(processed_img[0])  # Remove batch dimension
        processed_images = np.array(processed_images)
        
        # Điều chỉnh số epochs dựa vào số lượng mẫu
        num_samples = len(images)
        if num_samples < 5:
            # Khi có ít dữ liệu, tăng số epochs để học kỹ hơn
            adjusted_epochs = 30
            print(f"Phát hiện ít dữ liệu ({num_samples} mẫu), tăng số epochs lên {adjusted_epochs}")
        elif num_samples < 10:
            adjusted_epochs = 20
            print(f"Phát hiện ít dữ liệu ({num_samples} mẫu), tăng số epochs lên {adjusted_epochs}")
        else:
            adjusted_epochs = epochs
            
        # Print shapes for debugging
        print(f"Processed images shape: {processed_images.shape}")

        # ***** Thêm Debugging Nhãn cho bệnh *****
        print(f"DEBUG train(): Input disease_labels shape: {disease_labels.shape}")
        print(f"DEBUG train(): Current len(self.disease_classes): {len(self.disease_classes)}")
        print(f"DEBUG train(): Sample input disease labels (first 5):\n{disease_labels[:5]}")
        # *******************************
        
        # Ensure disease_labels are in the correct format and have the right number of classes
        if len(disease_labels.shape) == 1:
            print("DEBUG train(): Disease labels are 1D, converting to categorical.")
            disease_labels = tf.keras.utils.to_categorical(disease_labels, num_classes=len(self.disease_classes))
        elif disease_labels.shape[1] != len(self.disease_classes):
            print(f"DEBUG train(): Disease label shape mismatch! Input shape {disease_labels.shape[1]} != Current classes {len(self.disease_classes)}. Re-encoding.")
            # If labels have wrong number of classes, convert to indices first
            disease_indices = np.argmax(disease_labels, axis=1)
            print(f"DEBUG train(): Argmax indices from input disease labels (first 5): {disease_indices[:5]}")
            disease_labels = tf.keras.utils.to_categorical(disease_indices, num_classes=len(self.disease_classes))
            print(f"DEBUG train(): Re-encoded disease labels shape: {disease_labels.shape}")
            print(f"DEBUG train(): Sample re-encoded disease labels (first 5):\n{disease_labels[:5]}")
        else:
            print("DEBUG train(): Disease label shape matches class count. Using as is.")
        
        # Huấn luyện mô hình nhận diện loại cây nếu có nhãn
        if plant_labels is not None:
            print("Training Plant Type model...")
            
            # ***** Thêm Debugging Nhãn cho loại cây *****
            print(f"DEBUG train(): Input plant_labels shape: {plant_labels.shape}")
            print(f"DEBUG train(): Current len(self.plant_classes): {len(self.plant_classes)}")
            print(f"DEBUG train(): Sample input plant labels (first 5):\n{plant_labels[:5]}")
            # *******************************
            
            # Ensure plant_labels are in the correct format
            if len(plant_labels.shape) == 1:
                print("DEBUG train(): Plant labels are 1D, converting to categorical.")
                plant_labels = tf.keras.utils.to_categorical(plant_labels, num_classes=len(self.plant_classes))
            elif plant_labels.shape[1] != len(self.plant_classes):
                print(f"DEBUG train(): Plant label shape mismatch! Input shape {plant_labels.shape[1]} != Current classes {len(self.plant_classes)}. Re-encoding.")
                plant_indices = np.argmax(plant_labels, axis=1)
                print(f"DEBUG train(): Argmax indices from input plant labels (first 5): {plant_indices[:5]}")
                plant_labels = tf.keras.utils.to_categorical(plant_indices, num_classes=len(self.plant_classes))
                print(f"DEBUG train(): Re-encoded plant labels shape: {plant_labels.shape}")
                print(f"DEBUG train(): Sample re-encoded plant labels (first 5):\n{plant_labels[:5]}")
            else:
                print("DEBUG train(): Plant label shape matches class count. Using as is.")
            
            # Huấn luyện mô hình nhận diện loại cây
            plant_history = self.plant_type_model.fit(
                processed_images, 
                plant_labels,
                epochs=adjusted_epochs,
                batch_size=batch_size if num_samples >= batch_size else 1,
                validation_split=0.2 if len(processed_images) >= 5 else 0.0,
                verbose=1
            )
            print("Plant Type model training completed!")
        
        # Print shapes after processing for disease model
        print(f"Disease labels shape after processing: {disease_labels.shape}")
        
        # Kiểm tra số lượng mẫu
        if len(processed_images) < 5:  # Nếu có ít hơn 5 mẫu
            validation_split = 0.0  # Không sử dụng validation
        else:
            validation_split = 0.2  # Sử dụng 20% làm validation
            
        # Huấn luyện mô hình nhận diện bệnh
        disease_history = self.image_model.fit(
            processed_images, 
            disease_labels,
            epochs=adjusted_epochs,
            batch_size=batch_size if num_samples >= batch_size else 1,
            validation_split=validation_split,
            verbose=1
        )
        
        print("\nTraining Random Forest model...")
        # Convert disease_labels to class indices for Random Forest
        disease_indices = np.argmax(disease_labels, axis=1)
        print(f"DEBUG train(): Final indices for RandomForest (first 5): {disease_indices[:5]}")
        
        # Tuning Random Forest basad on sample size
        if num_samples < 5:
            # Với ít mẫu, tăng cường học
            n_estimators = 200
            max_depth = 3
        else:
            n_estimators = 100
            max_depth = None
            
        # Tạo RF mới với các tham số tối ưu
        self.env_model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            random_state=42,
            class_weight='balanced' if num_samples < 10 else None
        )
        
        self.env_model.fit(env_data, disease_indices)
        print(f"Training completed with {num_samples} samples!")
        
    def predict(self, img, env_data):
        """Dự đoán loại cây và bệnh dựa trên hình ảnh và dữ liệu môi trường"""
        try:
            # Kiểm tra nếu không có dữ liệu huấn luyện
            if len(self.training_data['env_data']) == 0:
                print("Chưa có dữ liệu huấn luyện. Vui lòng thêm dữ liệu huấn luyện trước khi dự đoán.")
                return {
                    'error': 'no_training_data',
                    'message': 'Chưa có dữ liệu huấn luyện. Vui lòng thêm dữ liệu huấn luyện trước khi dự đoán.'
                }
            
            # Tiền xử lý hình ảnh
            if isinstance(img, Image.Image):
                # Chuyển đổi PIL Image sang numpy array
                img = np.array(img)
            
            processed_img = self.preprocess_image(img)
            
            # Đảm bảo hình ảnh có đúng định dạng (224, 224, 3)
            if len(processed_img.shape) == 5:  # Nếu có thêm chiều batch
                processed_img = processed_img[0]  # Lấy phần tử đầu tiên
            if len(processed_img.shape) == 4:  # Nếu vẫn còn chiều batch
                processed_img = processed_img[0]  # Lấy phần tử đầu tiên
            
            # Tiền xử lý dữ liệu môi trường
            processed_env = self.preprocess_environment_data(*env_data)
            
            # Dự đoán loại cây
            plant_prediction = self.plant_type_model.predict(np.array([processed_img]))
            plant_type = self.plant_classes[np.argmax(plant_prediction[0])]
            plant_confidence = float(np.max(plant_prediction[0]))
            
            # Dự đoán bệnh từ hình ảnh
            disease_prediction = self.image_model.predict(np.array([processed_img]))
            disease_type = self.disease_classes[np.argmax(disease_prediction[0])]
            disease_confidence = float(np.max(disease_prediction[0]))
            
            # Dự đoán dựa trên dữ liệu môi trường
            try:
                env_prediction = self.env_model.predict_proba(np.array([processed_env]))
                env_disease = self.disease_classes[np.argmax(env_prediction[0])]
                env_confidence = float(np.max(env_prediction[0]))
            except Exception as e:
                print(f"Mô hình dự đoán môi trường chưa được huấn luyện đầy đủ: {str(e)}")
                env_disease = disease_type  # Sử dụng kết quả từ hình ảnh
                env_confidence = 0.0
            
            # Kiểm tra số lượng mẫu
            num_samples = len(self.training_data['env_data'])
            
            # Nếu có ít hơn 10 mẫu, ưu tiên kết quả từ hình ảnh
            if num_samples < 10:
                # Ưu tiên dự đoán từ hình ảnh khi còn ít dữ liệu huấn luyện
                final_disease = disease_type
                final_confidence = disease_confidence
                print(f"DEBUG: Ít dữ liệu ({num_samples} mẫu), ưu tiên kết quả từ hình ảnh: {disease_type}")
            else:
                # Có nhiều dữ liệu, kết hợp theo độ tin cậy
                final_disease = disease_type if disease_confidence > env_confidence else env_disease
                final_confidence = max(disease_confidence, env_confidence)
                print(f"DEBUG: Đủ dữ liệu ({num_samples} mẫu), kết hợp kết quả. Chọn: {final_disease}")
            
            # In thông tin debug
            print(f"DEBUG predict(): Dự đoán từ hình ảnh: {disease_type} ({disease_confidence:.2f})")
            print(f"DEBUG predict(): Dự đoán từ môi trường: {env_disease} ({env_confidence:.2f})")
            print(f"DEBUG predict(): Dự đoán cuối cùng: {final_disease} ({final_confidence:.2f})")
            
            return {
                'plant_type': plant_type,
                'plant_confidence': plant_confidence,
                'disease': final_disease,
                'disease_confidence': final_confidence,
                'image_disease': disease_type,
                'image_confidence': disease_confidence,
                'env_disease': env_disease,
                'env_confidence': env_confidence
            }
            
        except Exception as e:
            print(f"Lỗi khi dự đoán: {str(e)}")
            traceback.print_exc()
            return None 