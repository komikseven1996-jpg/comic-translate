from PySide6 import QtWidgets, QtCore
from ..dayu_widgets.label import MLabel

class AccountPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        account_layout = QtWidgets.QVBoxLayout(self)
        account_layout.setContentsMargins(20, 20, 20, 20)

        title_label = MLabel(self.tr("Account")).h3()
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        info_text = self.tr(
            "Comic Translate is ready to use. Configure your API keys "
            "in the Advanced settings section if needed."
        )
        info_label = MLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        account_layout.addWidget(title_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        account_layout.addSpacing(20)
        account_layout.addWidget(info_label)
        account_layout.addStretch(1)

        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
