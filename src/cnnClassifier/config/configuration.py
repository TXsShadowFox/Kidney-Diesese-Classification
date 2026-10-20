from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        if self.params.CLASSES != len(self.params.DATA_CLASSES):
            raise ValueError(f"CLASSES is {self.params.CLASSES} but DATA_CLASSES lists {len(self.params.DATA_CLASSES)} classes")
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_path=Path(config.source_path),
            train_dir=Path(config.train_dir),
            val_dir=Path(config.val_dir),
            test_dir=Path(config.test_dir),
            params_data_classes=list(self.params.DATA_CLASSES),
            params_validation_split=self.params.VALIDATION_SPLIT,
            params_test_split=self.params.TEST_SPLIT,
            params_seed=self.params.SEED
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        data_ingestion = self.config.data_ingestion
        params = self.params
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            classes_path=Path(training.classes_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(data_ingestion.train_dir),
            validation_data=Path(data_ingestion.val_dir),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=Path(self.config.training.trained_model_path),
            test_data=Path(self.config.data_ingestion.test_dir),
            mlflow_uri="https://dagshub.com/theBhavyaaggarwal81/Kidney-Disease-Classification-MLflow-DVC.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_positive_class=self.params.POSITIVE_CLASS
        )
        return eval_config