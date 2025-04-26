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
        
        # Khởi tạo mô hình Random Forest
        self.env_model = None
        
        # Khởi tạo danh sách dữ liệu training
        self.training_data = {
            'images': [],
            'env_data': [],
            'labels': [],
            'image_paths': []
        }
        
        # Tải dữ liệu training và mô hình nếu có
        if self.load_training_data():
            print("Đã tải mô hình và dữ liệu training thành công!")
        else:
            print("Khởi tạo mô hình mới...")
            # Khởi tạo mô hình mới
            self.image_model = self._build_cnn_model()
            self.env_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Tạo dữ liệu mẫu cho huấn luyện ban đầu
            n_samples = 100
            X_env = np.random.rand(n_samples, 4)  # 4 features: temperature, humidity, soil_moisture, light
            X_env[:, 0] = X_env[:, 0] * 15 + 20  # Temperature: 20-35°C
            X_env[:, 1] = X_env[:, 1] * 50 + 40  # Humidity: 40-90%
            X_env[:, 2] = X_env[:, 2] * 60 + 20  # Soil moisture: 20-80%
            X_env[:, 3] = X_env[:, 3] * 18000 + 2000  # Light: 2000-20000 lux
            
            # Tạo nhãn ngẫu nhiên cho dữ liệu mẫu
            y_env = np.random.randint(0, len(self.disease_classes), n_samples)
            
            # Huấn luyện mô hình Random Forest với dữ liệu mẫu
            self.env_model.fit(X_env, y_env)
            
            # Lưu dữ liệu mẫu
            self.training_data['env_data'] = X_env.tolist()
            # Chuyển đổi nhãn thành one-hot encoding
            labels = np.zeros((n_samples, len(self.disease_classes)))
            for i, label in enumerate(y_env):
                labels[i, label] = 1
            self.training_data['labels'] = labels.tolist()
            
            # Lưu dữ liệu và mô hình
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
    
    def add_training_data(self, image, image_path, env_data, label):
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
            
            # Thêm nhãn
            label_one_hot = np.zeros(len(self.disease_classes))
            label_one_hot[label] = 1
            self.training_data['labels'].append(label_one_hot.tolist())
            
            # Thêm đường dẫn ảnh mới
            self.training_data['image_paths'].append(new_image_path)
            
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
                y = np.array([self.training_data['labels'][i] for i in valid_indices])
                
                # Cập nhật lại training_data
                self.training_data['env_data'] = [self.training_data['env_data'][i] for i in valid_indices]
                self.training_data['labels'] = [self.training_data['labels'][i] for i in valid_indices]
                self.training_data['image_paths'] = [self.training_data['image_paths'][i] for i in valid_indices]
                
                # Huấn luyện mô hình
                self.train(X_img, X_env, None, y)  # Pass None for plant_labels since we're not using it
                
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
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Resize if needed
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))
        
        # Convert to float32 and normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        return image
    
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
        print("Training CNN model...")
        # Ensure images are properly preprocessed
        processed_images = []
        for img in images:
            processed_img = self.preprocess_image(img)
            processed_images.append(processed_img[0])  # Remove batch dimension
        processed_images = np.array(processed_images)
        
        # Print shapes for debugging
        print(f"Processed images shape: {processed_images.shape}")
        # print(f"Disease labels shape: {disease_labels.shape}") # Di chuyển xuống dưới
        # print(f"Number of disease classes: {len(self.disease_classes)}") # Di chuyển xuống dưới

        # ***** Thêm Debugging Nhãn *****
        print(f"DEBUG train(): Input disease_labels shape: {disease_labels.shape}")
        print(f"DEBUG train(): Current len(self.disease_classes): {len(self.disease_classes)}")
        # In một vài mẫu nhãn đầu vào (ví dụ: 5 mẫu đầu)
        print(f"DEBUG train(): Sample input labels (first 5):\n{disease_labels[:5]}")
        # *******************************
        
        # Ensure disease_labels are in the correct format and have the right number of classes
        if len(disease_labels.shape) == 1:
             # ***** Thêm Debugging *****
            print("DEBUG train(): Labels are 1D, converting to categorical.")
            # **************************
            disease_labels = tf.keras.utils.to_categorical(disease_labels, num_classes=len(self.disease_classes))
        elif disease_labels.shape[1] != len(self.disease_classes):
             # ***** Thêm Debugging *****
            print(f"DEBUG train(): Label shape mismatch! Input shape {disease_labels.shape[1]} != Current classes {len(self.disease_classes)}. Re-encoding.")
            # **************************
            # If labels have wrong number of classes, convert to indices first
            disease_indices = np.argmax(disease_labels, axis=1)
            # ***** Thêm Debugging *****
            print(f"DEBUG train(): Argmax indices from input labels (first 5): {disease_indices[:5]}")
            # **************************
            disease_labels = tf.keras.utils.to_categorical(disease_indices, num_classes=len(self.disease_classes))
            # ***** Thêm Debugging *****
            print(f"DEBUG train(): Re-encoded labels shape: {disease_labels.shape}")
            print(f"DEBUG train(): Sample re-encoded labels (first 5):\n{disease_labels[:5]}")
            # **************************
        else:
             # ***** Thêm Debugging *****
            print("DEBUG train(): Label shape matches class count. Using as is.")
             # **************************
        
        # Print shapes after processing
        print(f"Disease labels shape after processing: {disease_labels.shape}")
        
        # Kiểm tra số lượng mẫu
        if len(processed_images) < 5:  # Nếu có ít hơn 5 mẫu
            validation_split = 0.0  # Không sử dụng validation
        else:
            validation_split = 0.2  # Sử dụng 20% làm validation
            
        history = self.image_model.fit(
            processed_images, 
            disease_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        print("\nTraining Random Forest model...")
        # Convert disease_labels to class indices for Random Forest
        disease_indices = np.argmax(disease_labels, axis=1)
        # ***** Thêm Debugging *****
        print(f"DEBUG train(): Final indices for RandomForest (first 5): {disease_indices[:5]}")
        # **************************
        self.env_model.fit(env_data, disease_indices)
        print("Training completed!")
        
    def predict(self, img, env_data):
        """
        Dự đoán loại cây và bệnh từ ảnh và dữ liệu môi trường
        """
        try:
            # Preprocess image
            if len(img.shape) == 4:  # Nếu đã có batch dimension
                processed_img = img
            else:  # Nếu chưa có batch dimension
                processed_img = np.expand_dims(img, axis=0)
            
            # Normalize image if needed
            if processed_img.dtype != np.float32 or processed_img.max() > 1.0:
                processed_img = processed_img.astype('float32') / 255.0

            # Get predictions from CNN model
            cnn_pred = self.image_model.predict(processed_img, verbose=0)
            
            # Get predictions from Random Forest model for environment data
            if len(env_data.shape) == 1:
                env_data = env_data.reshape(1, -1)
            
            try:
                env_pred = self.env_model.predict_proba(env_data)
            except Exception as e:
                print(f"Lỗi khi dự đoán với Random Forest: {str(e)}")
                print("Huấn luyện lại mô hình Random Forest với dữ liệu mẫu...")
                # Tạo và huấn luyện lại với dữ liệu mẫu
                n_samples = 100
                X_env = np.random.rand(n_samples, 4)
                X_env[:, 0] = X_env[:, 0] * 15 + 20
                X_env[:, 1] = X_env[:, 1] * 50 + 40
                X_env[:, 2] = X_env[:, 2] * 60 + 20
                X_env[:, 3] = X_env[:, 3] * 18000 + 2000
                y_env = np.random.randint(0, len(self.disease_classes), n_samples)
                self.env_model.fit(X_env, y_env)
                env_pred = self.env_model.predict_proba(env_data)

            # Ensure env_pred has the same shape as disease_classes
            if env_pred.shape[1] != len(self.disease_classes):
                env_pred = np.random.rand(1, len(self.disease_classes))
                env_pred = env_pred / env_pred.sum(axis=1, keepdims=True)

            # Combine predictions for disease
            combined_pred = 0.7 * cnn_pred + 0.3 * env_pred
            
            return combined_pred
            
        except Exception as e:
            print(f"Lỗi trong quá trình dự đoán: {str(e)}")
            # Trả về dự đoán mặc định nếu có lỗi
            default_pred = np.zeros((1, len(self.disease_classes)))
            default_pred[0, 0] = 1  # Mặc định là "Khỏe mạnh"
            return default_pred 