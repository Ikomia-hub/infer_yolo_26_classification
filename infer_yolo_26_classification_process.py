import copy
import os

import torch

from ikomia import core, dataprocess, utils
from ultralytics import YOLO
from ultralytics import download


class InferYolo26ClassificationParam(core.CWorkflowTaskParam):
    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "yolo26m-cls"
        self.cuda = torch.cuda.is_available()
        self.input_size = 224
        self.update = False
        self.model_weight_file = ""

    def set_values(self, param_map):
        self.model_name = str(param_map["model_name"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.input_size = int(param_map["input_size"])
        self.model_weight_file = str(param_map["model_weight_file"])
        self.update = True

    def get_values(self):
        param_map = {
            "model_name": str(self.model_name),
            "cuda": str(self.cuda),
            "input_size": str(self.input_size),
            "update": str(self.update),
            "model_weight_file": str(self.model_weight_file)
        }
        return param_map


class InferYolo26Classification(dataprocess.CClassificationTask):
    def __init__(self, name, param):
        dataprocess.CClassificationTask.__init__(self, name)

        if param is None:
            self.set_param_object(InferYolo26ClassificationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.classes = None
        self.model = None
        self.half = False
        self.repo = "ultralytics/assets"
        self.version = "v8.4.0"

    def get_progress_steps(self):
        return 1

    def _load_model(self):
        param = self.get_param_object()
        self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
        self.half = True if param.cuda and torch.cuda.is_available() else False

        if param.model_weight_file:
            self.model = YOLO(param.model_weight_file)
        else:
            model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
            os.makedirs(model_folder, exist_ok=True)
            model_weights = os.path.join(str(model_folder), f"{param.model_name}.pt")

            if not os.path.isfile(model_weights):
                url = f"https://github.com/{self.repo}/releases/download/{self.version}/{param.model_name}.pt"
                download(url=url, dir=model_folder, unzip=True)

            self.model = YOLO(model_weights)

        categories = list(self.model.names.values())
        self.set_names(categories)
        param.update = False

    def init_long_process(self):
        self._load_model()
        super().init_long_process()

    def run(self):
        self.begin_task_run()

        param = self.get_param_object()

        img_input = self.get_input(0)
        src_image = img_input.get_image()

        if param.update:
            self._load_model()

        if self.is_whole_image_classification():
            results = self.model.predict(
                src_image,
                save=False,
                imgsz=param.input_size,
                half=self.half,
                device=self.device
            )

            classes_names = results[0].names
            probs = results[0].probs
            classes_idx = probs.top5

            t5_class_names = [classes_names[idx] for idx in classes_idx]
            t5_confidences = probs.top5conf.detach().cpu().numpy()
            confidence_str = [str(conf) for conf in t5_confidences]
            self.set_whole_image_results(t5_class_names, confidence_str)
        else:
            input_objects = self.get_input_objects()
            for obj in input_objects:
                roi_img = self.get_object_sub_image(obj)
                if roi_img is None:
                    continue

                results = self.model.predict(
                    roi_img,
                    save=False,
                    imgsz=param.input_size,
                    half=self.half,
                    device=self.device
                )

                probs = results[0].probs
                classes_idx = probs.top1
                confidence = probs.top1conf.detach().cpu().numpy()
                self.add_object(obj, classes_idx, float(confidence))

        self.emit_step_progress()
        self.end_task_run()


class InferYolo26ClassificationFactory(dataprocess.CTaskFactory):
    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        self.info.name = "infer_yolo_26_classification"
        self.info.short_description = "Inference with YOLO26 image classification models"
        self.info.path = "Plugins/Python/Classification"
        self.info.version = "1.0.0"
        self.info.min_ikomia_version = "0.16.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Jocher, G., Chaurasia, A., & Qiu, J"
        self.info.article = "YOLO by Ultralytics"
        self.info.journal = ""
        self.info.year = 2024
        self.info.license = "AGPL-3.0"
        self.info.documentation_link = "https://docs.ultralytics.com/"
        self.info.repository = "https://github.com/Ikomia-hub/infer_yolo_26_classification"
        self.info.original_repository = "https://github.com/ultralytics/ultralytics"
        self.info.keywords = "YOLO, YOLO26, classification, ultralytics, coco"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "CLASSIFICATION"
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        return InferYolo26Classification(self.info.name, param)
