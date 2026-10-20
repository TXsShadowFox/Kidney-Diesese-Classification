import io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path
from cnnClassifier.utils.common import load_json
from cnnClassifier.utils.preprocessing import preprocess_image


MODEL_PATH = os.path.join("artifacts", "training", "model.h5")
CLASSES_PATH = os.path.join("artifacts", "training", "classes.json")


class PredictionPipeline:
    def __init__(self):
        self.model = None
        self.classes = None
        self.model_mtime = None


    def get_model(self):
        # reload only when the file changed, e.g. after /train wrote a new model
        mtime = os.path.getmtime(MODEL_PATH)
        if self.model is None or mtime != self.model_mtime:
            self.model = load_model(MODEL_PATH)
            self.classes = list(load_json(Path(CLASSES_PATH)).classes)
            self.model_mtime = mtime
        return self.model


    def predict(self, image_bytes: bytes):
        model = self.get_model()

        # decode in memory and resize the same way the training generators do
        test_image = image.load_img(io.BytesIO(image_bytes), target_size = model.input_shape[1:3], interpolation="bilinear")
        test_image = preprocess_image(image.img_to_array(test_image))
        test_image = np.expand_dims(test_image, axis = 0)
        probabilities = model.predict(test_image)[0]
        result = int(np.argmax(probabilities))
        print(result)

        prediction = self.classes[result]
        # per-class softmax scores so the page can show how sure the model is
        scores = {name: float(p) for name, p in zip(self.classes, probabilities)}
        return [{ "image" : prediction, "probabilities" : scores}]
