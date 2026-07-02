from PySide6 import QtWidgets
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.combo_box import MComboBox

class ExportPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        batch_label = MLabel(self.tr("Automatic Mode")).h4()
        batch_note = MLabel(
            self.tr(
                "Selected exports are saved to comic_translate_<timestamp> in the same directory as the input file/archive."
            )
        ).secondary()
        batch_note.setWordWrap(True)
        self.raw_text_checkbox = MCheckBox(self.tr("Export Raw Text"))
        self.translated_text_checkbox = MCheckBox(self.tr("Export Translated text"))
        self.inpainted_image_checkbox = MCheckBox(self.tr("Export Inpainted Image"))

        # Output format
        format_label = MLabel(self.tr("Output Image Format")).h4()
        self.output_format_combo = MComboBox()
        self.output_format_combo.addItems([
            self.tr("JPEG"),
            self.tr("PNG"),
            self.tr("WebP"),
        ])
        format_hint = MLabel(
            self.tr("Format used when saving output images (cleaned, translated).")
        ).secondary()
        format_hint.setWordWrap(True)

        layout.addWidget(batch_label)
        layout.addWidget(batch_note)
        layout.addWidget(self.raw_text_checkbox)
        layout.addWidget(self.translated_text_checkbox)
        layout.addWidget(self.inpainted_image_checkbox)
        layout.addSpacing(12)
        layout.addWidget(format_label)
        layout.addWidget(self.output_format_combo)
        layout.addWidget(format_hint)

        layout.addStretch(1)
