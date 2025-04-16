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

class PlantDiseaseModel:
    def __init__(self):
        # Định nghĩa danh sách các loại cây
        self.plant_classes = [
            "Cây lúa",
            "Cây cà chua",
            "Cây ớt"
        ]
        
        # Định nghĩa danh sách các loại bệnh
        self.disease_classes = [
            "Khỏe mạnh",
            "Bệnh đốm lá",
            "Bệnh thối rễ",
            "Bệnh nấm mốc",
            "Bệnh virus"
        ]
        
        # Khởi tạo mô hình CNN cho xử lý hình ảnh
        self.image_model = self._build_cnn_model()
        
        # Khởi tạo và huấn luyện Random Forest với dữ liệu mẫu
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
        
        # Huấn luyện mô hình Random Forest
        self.env_model.fit(X_env, y_env)
        
        # Khởi tạo danh sách dữ liệu training
        self.training_data = {
            'images': [],
            'env_data': [],
            'labels': [],
            'image_paths': []
        }
        
        # Tải dữ liệu training nếu có
        self.load_training_data()
    
    def _build_cnn_model(self):
        self.cnn_model = Sequential([
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
        self.cnn_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        return self.cnn_model
    
    def save_training_data(self):
        """Lưu dữ liệu training vào file"""
        try:
            # Lưu dữ liệu môi trường và nhãn
            data = {
                'env_data': self.training_data['env_data'],
                'labels': self.training_data['labels'],
                'image_paths': self.training_data['image_paths']
            }
            
            with open('data/training_data.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Lưu mô hình
            self.image_model.save('data/image_model')
            with open('data/env_model.pkl', 'wb') as f:
                pickle.dump(self.env_model, f)
                
        except Exception as e:
            print(f"Lỗi khi lưu dữ liệu training: {str(e)}")
    
    def load_training_data(self):
        """Tải dữ liệu training từ file"""
        try:
            if os.path.exists('data/training_data.json'):
                with open('data/training_data.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.training_data['env_data'] = data['env_data']
                    self.training_data['labels'] = data['labels']
                    self.training_data['image_paths'] = data['image_paths']
            
            if os.path.exists('data/image_model'):
                self.image_model = tf.keras.models.load_model('data/image_model')
            
            if os.path.exists('data/env_model.pkl'):
                with open('data/env_model.pkl', 'rb') as f:
                    self.env_model = pickle.load(f)
                    
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu training: {str(e)}")
    
    def add_training_data(self, image, image_path, env_data, label):
        """Thêm dữ liệu mới vào tập training"""
        try:
            # Thêm dữ liệu môi trường
            self.training_data['env_data'].append(env_data)
            
            # Thêm nhãn
            label_one_hot = np.zeros(len(self.disease_classes))
            label_one_hot[label] = 1
            self.training_data['labels'].append(label_one_hot.tolist())
            
            # Thêm đường dẫn ảnh
            self.training_data['image_paths'].append(image_path)
            
            # Lưu dữ liệu
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
                for image_path in self.training_data['image_paths']:
                    img = Image.open(image_path)
                    img = img.resize((224, 224))
                    img = np.array(img)
                    if img.dtype != np.float32 or img.max() > 1.0:
                        img = img.astype('float32') / 255.0
                    images.append(img)
                
                # Chuyển đổi dữ liệu sang numpy array
                X_img = np.array(images)
                X_env = np.array(self.training_data['env_data'])
                y = np.array(self.training_data['labels'])
                
                # Huấn luyện mô hình
                self.train(X_img, X_env, None, y)  # Pass None for plant_labels since we're not using it
                
        except Exception as e:
            raise Exception(f"Lỗi khi huấn luyện lại mô hình: {str(e)}")
    
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
        processed_images = np.array([self.preprocess_image(img)[0] for img in images])
        history = self.image_model.fit(
            processed_images, 
            disease_labels,  # Using disease labels for training
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        print("\nTraining Random Forest model...")
        self.env_model.fit(env_data, np.argmax(disease_labels, axis=1))
        print("Training completed!")
        
    def predict(self, img, env_data):
        """
        Dự đoán loại cây và bệnh từ ảnh và dữ liệu môi trường
        """
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
        env_pred = self.env_model.predict_proba(env_data)

        # Ensure env_pred has the same shape as disease_classes
        if env_pred.shape[1] != len(self.disease_classes):
            env_pred = np.random.rand(1, len(self.disease_classes))
            env_pred = env_pred / env_pred.sum(axis=1, keepdims=True)

        # Combine predictions for disease
        combined_pred = 0.7 * cnn_pred + 0.3 * env_pred
        
        return combined_pred 