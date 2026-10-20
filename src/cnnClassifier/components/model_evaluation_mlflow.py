import numpy as np
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
from cnnClassifier.utils.preprocessing import preprocess_image


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config


    def _test_generator(self):

        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_image
        )

        self.test_generator = test_datagenerator.flow_from_directory(
            directory=self.config.test_data,
            shuffle=False,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)


    def _positive_class_index(self) -> int:
        class_indices = {name.lower(): index for name, index in self.test_generator.class_indices.items()}
        positive_class = self.config.params_positive_class.lower()
        if positive_class not in class_indices:
            raise ValueError(
                f"POSITIVE_CLASS '{self.config.params_positive_class}' is not one of "
                f"the classes {list(self.test_generator.class_indices)}"
            )
        return class_indices[positive_class]


    def _classification_metrics(self, probabilities: np.ndarray) -> dict:
        positive = self._positive_class_index()
        y_true = (self.test_generator.classes == positive).astype(int)
        y_pred = (np.argmax(probabilities, axis=1) == positive).astype(int)
        y_score = probabilities[:, positive]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        return {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            # recall is sensitivity: the share of positive (tumor) images the model catches
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "true_positive": int(tp),
            "false_negative": int(fn),
            "true_negative": int(tn),
            "false_positive": int(fp),
        }


    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._test_generator()
        # one pass over the test set; loss and accuracy are computed from the same predictions
        probabilities = self.model.predict(self.test_generator)
        y_true = self.test_generator.classes
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, probabilities))
        self.scores = {
            "loss": float(loss),
            "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == y_true)),
            **self._classification_metrics(probabilities),
        }
        self.save_score()

    def save_score(self):
        save_json(path=Path("scores.json"), data=self.scores)


    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(self.scores)
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
