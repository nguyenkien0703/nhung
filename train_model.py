import numpy as np
from model import PlantDiseaseModel
import cv2
import os
import random
from sklearn.model_selection import train_test_split

def create_plant_image(plant_type, disease_type):
    # Base colors for different plants
    plant_colors = {
        "Cây lúa": [120, 180, 70],      # Green for rice
        "Cây cà chua": [50, 150, 50],   # Dark green for tomato
        "Cây ớt": [100, 160, 60],       # Medium green for chili
        "Cây khoai tây": [80, 140, 40], # Darker green for potato
        "Cây dưa chuột": [90, 170, 50], # Light green for cucumber
        "Cây nho": [70, 130, 30]        # Deep green for grape
    }
    
    # Create a blank RGB image (224x224x3)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    base_color = plant_colors.get(plant_type, [100, 150, 50])
    
    # Fill with base color and add some random variation
    for c in range(3):
        image[:, :, c] = base_color[c] + np.random.randint(-20, 20, size=(224, 224))
    
    # Add disease patterns
    if disease_type != "Khỏe mạnh":
        # Add random spots or patterns based on disease type
        spots = np.random.rand(224, 224) > 0.95
        if disease_type == "Bệnh đốm lá":
            image[spots] = [139, 69, 19]  # Brown spots
        elif disease_type == "Bệnh thối rễ":
            image[spots] = [64, 64, 64]   # Dark spots
        elif disease_type == "Bệnh nấm mốc":
            image[spots] = [192, 192, 192] # White spots
        elif disease_type == "Bệnh virus":
            image[spots] = [255, 255, 0]   # Yellow spots
        elif disease_type == "Bệnh đạo ôn":
            image[spots] = [128, 0, 0]     # Dark red spots
        elif disease_type == "Bệnh bạc lá":
            image[spots] = [200, 200, 200] # Light gray spots
        elif disease_type == "Bệnh khô vằn":
            image[spots] = [139, 69, 19]   # Brown spots
        elif disease_type == "Bệnh mốc sương":
            image[spots] = [128, 128, 128] # Gray spots
        elif disease_type == "Bệnh thán thư":
            image[spots] = [139, 0, 0]     # Dark red spots
        elif disease_type == "Bệnh héo xanh":
            image[spots] = [0, 100, 0]     # Dark green spots
        elif disease_type == "Bệnh phấn trắng":
            image[spots] = [255, 255, 255] # White spots
        elif disease_type == "Bệnh sương mai":
            image[spots] = [128, 128, 128] # Gray spots
        elif disease_type == "Bệnh gỉ sắt":
            image[spots] = [139, 69, 19]   # Brown spots
        elif disease_type == "Bệnh ghẻ":
            image[spots] = [128, 0, 0]     # Dark red spots
            
    return image

def create_sample_data(num_samples=300, plant_classes=None, disease_classes=None):
    if plant_classes is None:
        plant_classes = ["Cây lúa", "Cây cà chua", "Cây ớt", "Cây khoai tây", "Cây dưa chuột", "Cây nho"]
    if disease_classes is None:
        disease_classes = ["Khỏe mạnh", "Bệnh đốm lá", "Bệnh thối rễ", "Bệnh nấm mốc", "Bệnh virus", 
                         "Bệnh đạo ôn", "Bệnh bạc lá", "Bệnh khô vằn", "Bệnh mốc sương", "Bệnh thán thư",
                         "Bệnh héo xanh", "Bệnh phấn trắng", "Bệnh sương mai", "Bệnh gỉ sắt", "Bệnh ghẻ"]

    # Initialize arrays
    images = np.zeros((num_samples, 224, 224, 3), dtype=np.uint8)
    env_data = np.zeros((num_samples, 4))
    plant_labels = np.zeros((num_samples, len(plant_classes)))  # One-hot encoded plant labels
    disease_labels = np.zeros((num_samples, len(disease_classes)))  # One-hot encoded disease labels

    for i in range(num_samples):
        # Select random plant and disease type
        plant_idx = random.randint(0, len(plant_classes) - 1)
        disease_idx = random.randint(0, len(disease_classes) - 1)
        
        # Generate image
        images[i] = create_plant_image(plant_classes[plant_idx], disease_classes[disease_idx])
        
        # Generate environmental data
        temp = random.uniform(20, 35)  # Temperature in Celsius
        humidity = random.uniform(40, 90)  # Humidity percentage
        soil_moisture = random.uniform(20, 80)  # Soil moisture percentage
        light = random.uniform(2000, 20000)  # Light intensity in lux
        env_data[i] = [temp, humidity, soil_moisture, light]
        
        # One-hot encode labels
        plant_labels[i, plant_idx] = 1
        disease_labels[i, disease_idx] = 1

    return images, env_data, plant_labels, disease_labels

def main():
    print("Tạo dữ liệu mẫu...")
    model = PlantDiseaseModel()  # Khởi tạo model trước
    X_img, X_env, y_plant, y_disease = create_sample_data(300, model.plant_classes, model.disease_classes)

    # Split data into training and testing sets
    X_img_train, X_img_test = X_img[:250], X_img[250:]
    X_env_train, X_env_test = X_env[:250], X_env[250:]
    y_plant_train, y_plant_test = y_plant[:250], y_plant[250:]
    y_disease_train, y_disease_test = y_disease[:250], y_disease[250:]

    print("Khởi tạo và huấn luyện mô hình...")
    
    # Train the model
    print("Huấn luyện mô hình...")
    model.train(X_img_train, X_env_train, y_plant_train, y_disease_train)
    
    # Đảm bảo lưu mô hình sau khi huấn luyện
    print("Lưu mô hình...")
    model.save_training_data()

    # Make predictions on test data
    print("\nKiểm tra mô hình với dữ liệu test...")
    for i in range(5):
        disease_pred = model.predict(X_img_test[i:i+1], X_env_test[i:i+1])
        
        actual_plant = np.argmax(y_plant_test[i])
        actual_disease = np.argmax(y_disease_test[i])
        pred_disease = np.argmax(disease_pred[0])
        
        print(f"\nMẫu thử {i+1}:")
        print(f"Thực tế: Loại cây - {model.plant_classes[actual_plant]}, "
              f"Tình trạng - {model.disease_classes[actual_disease]}")
        print(f"Dự đoán: Tình trạng - {model.disease_classes[pred_disease]}")
        print(f"Xác suất các bệnh:")
        for j, prob in enumerate(disease_pred[0]):
            print(f"  {model.disease_classes[j]}: {prob:.2%}")

if __name__ == "__main__":
    main() 