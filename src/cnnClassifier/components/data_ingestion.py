import os
import posixpath
import random
import shutil
import zipfile
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import (DataIngestionConfig)


# the extensions Keras' flow_from_directory picks up
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".tif", ".tiff"}


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def ingest_data(self):
        """
        Reads the DATA_CLASSES folders from source_path (the Kaggle CT kidney
        dataset zip, or the folder it was extracted to) and copies them into
        train/val/test folders. Other classes in the dataset are ignored.
        """
        source = self.config.source_path
        if not os.path.exists(source):
            raise FileNotFoundError(
                f"Dataset not found at '{source}'. Download the Kaggle CT kidney dataset "
                "and set data_ingestion.source_path in config/config.yaml"
            )
        logger.info(f"Reading classes {list(self.config.params_data_classes)} from {source}")

        if os.path.isdir(source):
            paths = [
                os.path.relpath(os.path.join(root, f), source).replace(os.sep, "/")
                for root, _, files in os.walk(source) for f in files
            ]
            images = self._find_images(paths)
            self._write_splits(images, lambda path: open(os.path.join(source, path), "rb"))
        else:
            with zipfile.ZipFile(source) as zip_ref:
                images = self._find_images(zip_ref.namelist())
                self._write_splits(images, zip_ref.open)


    def _find_images(self, paths: list) -> dict:
        """
        Groups image paths ("/"-separated) by the folder they sit in. Only folders
        named like a DATA_CLASSES entry count (case-insensitive), and each class
        must come from exactly one folder, so no image can land in two splits.
        """
        wanted = {name.lower(): name for name in self.config.params_data_classes}
        found = {}
        for path in paths:
            folder, file_name = posixpath.split(path)
            if "__MACOSX" in folder.split("/") or file_name.startswith("."):
                continue
            class_name = wanted.get(posixpath.basename(folder).lower())
            if class_name and posixpath.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS:
                found.setdefault(class_name, {}).setdefault(folder, []).append(path)

        images = {}
        for class_name in self.config.params_data_classes:
            folders = found.get(class_name, {})
            if len(folders) != 1:
                raise ValueError(
                    f"Expected exactly one '{class_name}' image folder in {self.config.source_path}, "
                    f"found {len(folders)}: {sorted(folders)}"
                )
            images[class_name] = sorted(next(iter(folders.values())))
        return images


    def _write_splits(self, images: dict, open_image):
        """
        Splits each class with a fixed seed, so every run gets the same split.
        The test folder is only used by the evaluation stage.
        """
        split_dirs = {
            "train": self.config.train_dir,
            "val": self.config.val_dir,
            "test": self.config.test_dir,
        }
        for split_dir in split_dirs.values():
            shutil.rmtree(split_dir, ignore_errors=True)

        rng = random.Random(self.config.params_seed)
        for class_name in sorted(images):
            files = list(images[class_name])
            rng.shuffle(files)

            n_test = round(len(files) * self.config.params_test_split)
            n_val = round(len(files) * self.config.params_validation_split)
            splits = {
                "test": files[:n_test],
                "val": files[n_test:n_test + n_val],
                "train": files[n_test + n_val:],
            }

            for split_name, split_files in splits.items():
                if not split_files:
                    raise ValueError(f"{split_name} split has no images for class '{class_name}' ({len(files)} found)")
                target_dir = os.path.join(split_dirs[split_name], class_name)
                os.makedirs(target_dir, exist_ok=True)
                for path in split_files:
                    with open_image(path) as src, open(os.path.join(target_dir, posixpath.basename(path)), "wb") as dst:
                        shutil.copyfileobj(src, dst)

            logger.info(f"{class_name}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test images")
