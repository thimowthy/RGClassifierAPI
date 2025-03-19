import os
import cv2
import fitz
import rembg
import shutil
import numpy as np
from PIL import Image
from pathlib import Path


def extractImagesFromPdf(filepath, outputFolder) -> list[str]:

    pdf = fitz.open(filepath)
    filename = os.path.basename(filepath).replace(".pdf", "")
    outputPaths = []

    for pageNum, page in enumerate(pdf, start=1):
        img = page.get_pixmap()
        outputPath = os.path.join(outputFolder, f"{filename}_{pageNum}.png")
        img.save(outputPath)
        outputPaths.append(f"{filename}_{pageNum}.png")

    return outputPaths


def removeBackground(filepath, outputFolder) -> str:

    os.makedirs(outputFolder, exist_ok=True)
    filename = os.path.basename(filepath)
    inputImage = Image.open(filepath)
    inputArray = np.array(inputImage)
    outputArray = rembg.remove(inputArray)
    outputImage = Image.fromarray(outputArray)
    outputPath = os.path.join(outputFolder, filename.replace(".jpg", ".png").replace(".jpeg", ".png"))
    outputImage.save(outputPath)
    
    return outputPath


def extractRoiFromImage(filepath, outputFolder):

    croppedImgPaths = []
    file = os.path.basename(filepath)
    croppedImages = cropImageRois(filepath)
    
    for i, croppedImage in enumerate(croppedImages, start=1):
        outputPath = os.path.join(outputFolder, file.replace('.', f"_{i}."))
        cv2.imwrite(outputPath, croppedImage)
        croppedImgPaths.append(outputPath)

    return croppedImgPaths


def cropImageRois(filepath):
    try:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        if img.shape[-1] == 4:
            alpha = img[:, :, 3]
            bg_mask = alpha == 0

            img = img[:, :, :3]
            img[bg_mask] = [255, 255, 255]

        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshGray = cv2.threshold(grayImg, thresh=200, maxval=255, type=cv2.THRESH_BINARY_INV)  # Threshold alto para remover fundo claro

        contours, _ = cv2.findContours(threshGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        X, Y = grayImg.shape
        
        imgArea = X*Y
        croppedImages = []

        for contour in contours:
            contourArea = cv2.contourArea(contour)
            occupiedArea = contourArea/imgArea # % da área coberta pelo contorno
            areaThreshold = 0.05

            if occupiedArea >= areaThreshold:

                convexHull = cv2.convexHull(contour)
                mask = np.zeros_like(grayImg)
                cv2.fillConvexPoly(mask, convexHull, 255)
                mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                cropped_img = cv2.bitwise_and(img, mask_3channel)
                x, y, w, h = cv2.boundingRect(convexHull)
                
                cropped_img = cropped_img[y:y+h, x:w+x]
                croppedImages.append(cropped_img)

        return croppedImages
    except Exception as e:
        print(e)
        print(f"Erro ao processar imagem: {filepath}")


def extractImages(assetsFolder: str | Path = r"C:\Users\dannilo.costa\Desktop\ClassificadorRG\buildClassifier\data"):

    BgFolder = "background"
    noBgFolder = "backgroundRemoved"
    
    croppedBg = "croppedBg"
    croppedNoBg = "croppedNoBg"

    folders = [BgFolder, noBgFolder, croppedBg, croppedNoBg]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if os.path.isfile(path):
                os.remove(path)
    
    bgMapping = dict()
    noBgMapping = dict()

    allowedExtensions = [".png", ".jpg", ".jpeg", ".pdf"]

    for file in os.listdir(assetsFolder):
        bgMapping[file] = []
        pages = [file]

        if file.endswith(".pdf"):
            filepath = os.path.join(assetsFolder, file)
            pages = extractImagesFromPdf(filepath, assetsFolder)

        for page in pages:
            filename = os.path.basename(page)
            if any([file.endswith(extension) for extension in allowedExtensions]):        
                filepath = os.path.join(assetsFolder, filename)
                newfilepath = os.path.join(BgFolder, filename)
                shutil.copyfile(filepath, newfilepath)
                bgMapping[file].append(newfilepath)

    for file, paths in bgMapping.items():
        noBgMapping[file] = []
        for path in paths:
            noBgMapping[file].append(removeBackground(path, noBgFolder))
    
    for file, paths in bgMapping.items():
        bgMapping[file] = [extractRoiFromImage(path, croppedBg) for path in paths]
    
    
    for file, paths in noBgMapping.items():
        noBgMapping[file] = [extractRoiFromImage(path, croppedNoBg) for path in paths]
    
    
    return bgMapping, noBgMapping


if __name__=="__main__":
    path = r"C:\Users\dannilo.costa\Desktop\ClassificadorRG\MaskRCNN\backgroundRemoved\2010_1.png"
    extractRoiFromImage(path, "assets")


