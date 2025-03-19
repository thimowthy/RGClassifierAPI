import os
import cv2
import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from preprocessing import preprocessImage
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def CNN(input_shape=(128, 128, 3)):
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model


def load_data(folder_dir):
    categories = os.listdir(folder_dir)
    label_map = {category: i for i, category in enumerate(categories)}
    X, y = [], []
    
    for category in categories:
        class_dir = os.path.join(folder_dir, category)
        label = label_map[category]
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (128, 128))
            X.append(img)
            y.append(label)
    
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_and_evaluate(model_fn, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    model = model_fn()
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Predições e métricas detalhadas
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model




# folder_dir = r"C:\Users\dannilo.costa\Desktop\ClassificadorRG\MaskRCNN\final\processed"
# X_train, X_test, y_train, y_test = load_data(folder_dir)

# print("Training Optimized CNN")
# optimized_model = train_and_evaluate(CNN, X_train, y_train, X_test, y_test)

# optimized_model.save("modelo_classificador_rg.h5")

# import cv2
# import numpy as np

# # Load the saved model
# from tf_keras.models import load_model
# model = load_model("modelo_classificador_rg.h5")


def predictImage(img_path, model):

    preProcessedImg = preprocessImage(img_path)
    
    prediction = model.predict(preProcessedImg)

    pred_class = (prediction > 0.5).astype("int32")
    
    return pred_class[0][0]


# # Path to the image you want to predict
# img_path = r"C:\Users\dannilo.costa\Desktop\ClassificadorRG\MaskRCNN\final\not_rg\1989 V_1.jpg"

# img_path = r"C:\Users\dannilo.costa\Desktop\ClassificadorRG\MaskRCNN\final\rg\95 MS_1.png"

# predicted_class = predictImage(img_path, model)
# print(f"Predicted Class: {predicted_class}")
