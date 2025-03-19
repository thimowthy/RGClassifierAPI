import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from extraction import extractImages
from classification import predictImage
from flask import Flask, request, jsonify
from tf_keras.models import load_model


app = Flask(__name__)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model("modelo_classificador_rg.h5")

@app.route("/classificarDocumentos", methods=["POST"])
def classify():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    filesPrediction = dict()

    for file in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, file)
        if os.path.isfile(path):
            os.remove(path)

    for file in files:
        if file.filename == "":
            continue

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        filesPrediction[file.filename] = []

    bgMapping, noBgMapping = extractImages(UPLOAD_FOLDER)
    
    bgPredictions = dict()
    noBgPredictions = dict()

    for file in filesPrediction:

        bgPredictions[file] = all([ predictImage(imgRoi, model) for imgRois in bgMapping[file] for imgRoi in imgRois ])
        noBgPredictions[file] = all([ predictImage(imgRoi, model) for imgRois in noBgMapping[file] for imgRoi in imgRois ])

        filesPrediction[file] = bgPredictions[file] or noBgPredictions[file]
    
    return jsonify(filesPrediction)

if __name__ == "__main__":
    app.run(debug=True)