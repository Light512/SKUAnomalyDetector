import os
import numpy as np
from PIL import Image
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from src.config import image_processor, model, DEVICE


def extract_features(image):
    inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def load_images_and_extract_features(image_root):
    image_paths = []
    features = []

    for class_folder in os.listdir(image_root):
        class_folder_path = os.path.join(image_root, class_folder)
        if os.path.isdir(class_folder_path):
            print(f"Processing class: {class_folder}")
            for image_name in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_name)
                if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        image = Image.open(image_path).convert("RGB")
                        feature = extract_features(image)
                        features.append(feature)
                        image_paths.append(image_path)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
    return np.array(features), image_paths


def detect_anomalies(features, image_paths, contamination=0.05):
    features = StandardScaler().fit_transform(features)
    features = PCA(n_components=min(34, features.shape[1])).fit_transform(features)

    print("Running Isolation Forest for anomaly detection...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    labels = iso_forest.fit_predict(features)

    results = []
    for image_path, label in zip(image_paths, labels):
        class_folder = os.path.basename(os.path.dirname(image_path))
        results.append({
            'image_path': image_path,
            'class': class_folder,
            'anomaly': 1 if label == -1 else 0
        })

    return results
