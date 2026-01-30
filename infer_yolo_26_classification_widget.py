from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_yolo_26_classification.infer_yolo_26_classification_process import InferYolo26ClassificationParam

from PyQt5.QtWidgets import *
from torch.cuda import is_available


class InferYolo26ClassificationWidget(core.CWorkflowTaskWidget):
    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferYolo26ClassificationParam()
        else:
            self.parameters = param

        self.grid_layout = QGridLayout()

        self.check_cuda = pyqtutils.append_check(
            self.grid_layout, "Cuda", self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())

        self.combo_model = pyqtutils.append_combo(self.grid_layout, "Model name")
        self.combo_model.addItem("yolo26n-cls")
        self.combo_model.addItem("yolo26s-cls")
        self.combo_model.addItem("yolo26m-cls")
        self.combo_model.addItem("yolo26l-cls")
        self.combo_model.addItem("yolo26x-cls")
        self.combo_model.setCurrentText(self.parameters.model_name)

        custom_weight = bool(self.parameters.model_weight_file)
        self.check_cfg = QCheckBox("Custom model")
        self.check_cfg.setChecked(custom_weight)
        self.grid_layout.addWidget(self.check_cfg, self.grid_layout.rowCount(), 0, 1, 2)
        self.check_cfg.stateChanged.connect(self.on_custom_weight_changed)

        self.label_hyp = QLabel("Model weight (.pt)")
        self.browse_weight_file = pyqtutils.BrowseFileWidget(
            path=self.parameters.model_weight_file,
            tooltip="Select file",
            mode=QFileDialog.ExistingFile
        )
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_hyp, row, 0)
        self.grid_layout.addWidget(self.browse_weight_file, row, 1)

        self.label_hyp.setVisible(custom_weight)
        self.browse_weight_file.setVisible(custom_weight)

        self.spin_input_size = pyqtutils.append_spin(
            self.grid_layout,
            "Input size",
            self.parameters.input_size
        )

        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)
        self.set_layout(layout_ptr)

    def on_custom_weight_changed(self, int):
        self.label_hyp.setVisible(self.check_cfg.isChecked())
        self.browse_weight_file.setVisible(self.check_cfg.isChecked())

    def on_apply(self):
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.input_size = self.spin_input_size.value()
        if self.check_cfg.isChecked():
            self.parameters.model_weight_file = self.browse_weight_file.path
        self.parameters.update = True

        self.emit_apply(self.parameters)


class InferYolo26ClassificationWidgetFactory(dataprocess.CWidgetFactory):
    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        self.name = "infer_yolo_26_classification"

    def create(self, param):
        return InferYolo26ClassificationWidget(param, None)
