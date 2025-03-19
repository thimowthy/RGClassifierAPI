import os
import cv2
import numpy as np


def preprocessImage(filepath: str):

    img = cv2.imread(filepath)
    
    if img is None:
        print("Error: Image not found.")
        return None
    
    resizedImg = cv2.resize(img, dsize=(128, 128))
    resizedImg = resizedImg.astype(np.float32) / 255.0
    resizedImg = np.expand_dims(resizedImg, axis=0)
    
    return resizedImg


def rotate_image(image, angle):

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    
    return rotated


if __name__=="__main__":
    
    for file in os.listdir(r"C:\Users\dannilo.costa\Desktop\ClassificadorRG\MaskRCNN\final\rg"):
        filepath = os.path.join(r"C:\Users\dannilo.costa\Desktop\ClassificadorRG\MaskRCNN\final\rg", file)
        img = cv2.imread(filepath)
        resized_img = cv2.resize(img, dsize=(128, 128))

        output_dir = r"C:\Users\dannilo.costa\Desktop\ClassificadorRG\MaskRCNN\final\processed\rg"

        for i in range(3):
            angle = np.random.uniform(0, 360)
            rotated_img = rotate_image(resized_img, angle)
            cv2.imwrite(os.path.join(output_dir, f"rotated_{i}_{file}".replace(".jpg", ".png")), rotated_img)