# JSON STRUCTURE
# {
#     "first_path": [
#         {
#             "row": 0,
#             "col": 0,
#             "label": "*"
#         },
#         {
#             "row": 0,
#             "col": 0,
#             "label": "*"
#         },
#         {
#             "row": 0,
#             "col": 0,
#             "label": "*"
#         },
#         {
#             "row": 0,
#             "col": 0,
#             "label": "*"
#         }
#     ]
# }


from VideoProcess.test_conv import CornerLabeler

from PyQt5.QtGui import QIntValidator, QMouseEvent, QImage, QPixmap, QWheelEvent, QKeySequence, QFont, QFocusEvent
from PyQt5.QtCore import (Qt, QSize, QTimer, QEvent, pyqtSignal, pyqtSlot, QModelIndex, QPoint)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStatusBar, QDockWidget, QDialog,
    QToolBar, QFileDialog, QMessageBox, QInputDialog,
    QLineEdit, QSlider,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QLayout, QPushButton, QSizePolicy, QLabel,
    QMenu, QAction, QFrame, QListWidget, QAbstractItemView, QTableWidgetSelectionRange
)

import pyqtgraph as pg

import numpy as np

import os
import sys
import pathlib
import json
from functools import partial

from typing import Tuple, Dict, List, Any, Union

from datetime import datetime


class Block():
    def __init__(self, label: str = '**', row: int = 0, col: int = 0):
        self.label = label
        self.row = row
        self.col = col

    def on_label_changed(self, text: str):
        self.label = text
        print(f'> Edited block [{self.row}][{self.col}] to "{self.label}"')


class Patch():
    def __init__(self, name: str, blocks: list[list[Block]]):
        self.name = name
        self.blocks = blocks

    @classmethod
    def new(self, name: str = 'NewPatch', rows: int = 3, cols: int = 3):
        return self(name, [[Block(row=x, col=y) for y in range(cols)] for x in range(rows)])

    def duplicate(self):
        return Patch(self.name, self.blocks)

    def rows(self):
        return len(self.blocks)

    def cols(self):
        return len(self.blocks[0]) if self.rows() > 0 else 0

    def setCols(self, val: int, asdelta: bool = False):
        if not asdelta:
            val = val - self.cols()
        if val > 0:
            for i in range(val):
                for col in self.blocks:
                    col.append(Block(row=self.rows(), col=len(col)))
        elif val < 0:
            for i in range(abs(val)):
                for col in self.blocks:
                    col.pop(len(col)-1)

    def setRows(self, val: int, asdelta: bool = False):
        if not asdelta:
            val = val - self.rows()
        if val > 0:
            for i in range(val):
                self.blocks.append([Block(row=self.rows(), col=i) for i in range(len(self.blocks[0]))])
        elif val < 0:
            for i in range(abs(val)):
                self.blocks.pop(self.rows()-1)


class CustomLine(QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mousePressEvent(self, a0):
        super().mousePressEvent(a0)
        self.selectAll()


class PatchWidget(QWidget):
    sigPatchesChanged = pyqtSignal(name='patchesChanged')
    sigPatchSelected = pyqtSignal(int, name='patchSelected')

    help_msg = '''
    1. Open a video/image file using File -> Open (Ctrl+O)
    2. Left-click in the image to place a keypoint, Ctrl+left-click to remove a keypoint
    3. Select keypoints in the KeyPointView, then right-click -> remove to remove a keypoint/keypoints
    4. File -> Load/Save to interact with an annotation file
    5. File -> Export to export keypoints and keypointed images into a folder
        '''

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.zoom = 1.0

        self.patches: list[Patch] = []
        self.selectedPatch: Patch = None

        self.sigPatchSelected.connect(self.on_patch_selected)

        layout = QGridLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.setLayout(layout)

    # def save_patches(self, fn):

        # patches = [for block in ]

        # json_string = json.dumps(patches, separators=(',', ":"))  # Compact JSON structure
        # open(fn, "w+", 1).write(json_string)

    def clear_layout(self):
        while self.layout().count() > 0:
            item = self.layout().itemAt(0)
            widget = item.widget()
            if widget is None:
                self.layout().removeItem(item)
            else:
                self.layout().removeWidget(widget)

    def update(self):
        if self.selectedPatch:
            self.clear_layout()

            title = QPushButton()
            title.setText(self.selectedPatch.name)
            title.setFont(QFont('Consolas', 48))
            title.setDisabled(True)
            self.layout().addWidget(title, 0, 0, 1, self.selectedPatch.cols())
            # title.move(QPoint(0,-200))

            for x, col in enumerate(self.selectedPatch.blocks):
                for y, block in enumerate(col):
                    self.layout().addWidget(PatchWidget.getLineWidget(block), x+1, y)

            add_col_btn = QPushButton()
            # add_col_btn.clicked.connect() // todo
            add_col_btn.setText('+')
            add_col_btn.setFont(QFont('Consolas', 32))
            add_col_btn.setFixedWidth(35)
            add_col_btn.setSizePolicy(1, 3)
            self.layout().addWidget(add_col_btn, 1, self.selectedPatch.cols(), self.selectedPatch.rows(), 1)
            rem_col_btn = QPushButton()
            rem_col_btn.setText('-')
            rem_col_btn.setFont(QFont('Consolas', 32))
            rem_col_btn.setFixedWidth(35)
            rem_col_btn.setSizePolicy(1, 3)
            self.layout().addWidget(rem_col_btn, 1, self.selectedPatch.cols()+1, self.selectedPatch.rows(), 1)

            add_row_btn = QPushButton()
            add_row_btn.setText('+')
            add_col_btn.setFont(QFont('Consolas', 32))
            add_row_btn.setFixedHeight(35)
            self.layout().addWidget(add_row_btn, self.selectedPatch.rows()+1, 0, 1, self.selectedPatch.cols())
            rem_row_btn = QPushButton()
            rem_row_btn.setText('-')
            rem_row_btn.setFont(QFont('Consolas', 32))
            rem_row_btn.setFixedHeight(35)
            self.layout().addWidget(rem_row_btn, self.selectedPatch.rows()+2, 0, 1, self.selectedPatch.cols())

    @pyqtSlot(int)
    def on_patch_selected(self, index: int):
        patch = self.patches[index]
        if self.selectedPatch != patch:
            self.selectedPatch = patch
            print(f'> Selected ({index}) "{patch.name}" | {patch.rows()}x{patch.cols()} blocks')

    @staticmethod
    def getLineWidget(block: Block):
        line = CustomLine(None)
        line.setText(block.label)
        line.textChanged[str].connect(block.on_label_changed)

        # line.editingFinished.connect(line.clearFocus)
        line.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        line.setMaxLength(2)
        line.setFixedWidth(100)

        f = QFont('Consolas', 36)
        f.setCapitalization(True)
        f.setKerning(True)
        f.setBold(True)
        line.setFont(f)

        return line


class Window(QMainWindow):
    def __init__(self):
        super().__init__(parent=None)
        self.setWindowTitle("Patches Tool")

        self.main_widget = PatchWidget()
        self.setCentralWidget(self.main_widget)

        # Patches List
        self._createPatchesViews()

        # Menu
        self._createMenu()

        # Status bar
        self.status = QStatusBar()
        self.status.showMessage('Init')
        self.setStatusBar(self.status)

        self.patches_file = None

    def add_patch_ask_user(self):
        self.add_patch(ask_user=True)

    def add_patch_default(self):
        self.add_patch(None, select=True)

    def add_patch(self, patch: Patch = None, insert_index: int = -1, select: bool = False, ask_user: bool = False):

        if patch == None:
            patch = Patch.new()
            if ask_user:
                name, ret = QInputDialog.getText(self, 'Patch Name', 'Patch Name', text='NewPatch')
                if not ret:
                    return
                patch.name = name

                rows, ret = QInputDialog.getInt(self, 'Rows', 'Rows', 3, min=1)
                if not ret:
                    return

                cols, ret = QInputDialog.getInt(self, 'Cols', 'Cols', 3, min=1)
                if not ret:
                    return

                patch.setRows(rows)
                patch.setCols(cols)

        self.main_widget.patches.append(patch) if insert_index == -1 else self.main_widget.patches.insert(insert_index, patch)

        print(f'> Added patch "{patch.name}"')

        self.main_widget.sigPatchesChanged.emit()
        self.main_widget.update()

        if select:
            self.table.selectRow(self.table.rowCount()-1) if insert_index == -1 else self.table.selectRow(insert_index+1)

    def duplicate_patch(self):
        idx = self.table.currentIndex().row()

        self.add_patch(self.main_widget.patches[idx].duplicate(), idx, True)

    def remove_patch(self):
        idx = self.table.currentIndex().row()
        self.table.removeRow(idx)

        self.main_widget.patches.remove(self.main_widget.patches[idx])

        self.main_widget.sigPatchesChanged.emit()
        self.main_widget.update()

    def patch_selected(self):
        idx = self.table.currentIndex().row()

        self.main_widget.sigPatchSelected.emit(idx)
        self.main_widget.update()

    def patch_changed(self):
        if len(self.table.selectedIndexes()) == 0:
            return
        idx = self.table.indexFromItem(self.table.currentItem()).row()
        self.main_widget.patches[idx].name = self.table.item(idx, 0).value
        self.main_widget.patches[idx].setRows(self.table.item(idx, 1).value)
        self.main_widget.patches[idx].setCols(self.table.item(idx, 2).value)

        self.main_widget.sigPatchesChanged.emit()
        self.main_widget.update()

    def _createPatchesViews(self):
        # List of patches
        self.table = pg.TableWidget(editable=True, sortable=True)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # self.table.setHorizontalHeader(['Name', 'Rows', 'Cols'])
        self.table.setFont(QFont('Consolas', 24))
        self.table.itemSelectionChanged.connect(self.patch_selected)
        self.table.model().dataChanged.connect(self.patch_changed)

        menu: QMenu = self.table.contextMenu
        menu.addSeparator()

        menu.addAction('Add').triggered.connect(self.add_patch_default)
        menu.addAction('Duplicate').triggered.connect(self.duplicate_patch)
        menu.addAction('Remove').triggered.connect(self.remove_patch)

        table_widget = QDockWidget('List of Patches')
        table_widget.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        table_widget.setMinimumWidth(600)
        table_widget.setWidget(self.table)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, table_widget)

        def update_patches():
            curr_list = self.table.items
            new_list = self.main_widget.patches

            if curr_list != new_list:
                data = [[p.name, p.rows(), p.cols()] for p in new_list]
                self.table.setData(data)

        self.main_widget.sigPatchesChanged.connect(update_patches)

    def _createMenu(self):
        # File menu
        menu = self.menuBar().addMenu("&File")
        menu.addAction("&Open...", self.open, shortcut='Ctrl+O')
        menu.addSeparator()
        menu.addAction("&Save...", self.save, shortcut='Ctrl+S')
        menu.addAction("Save As...", self.saveAs)
        menu.addSeparator()
        menu.addAction("&Exit", self.close)

        # Edit menu
        menu = self.menuBar().addMenu("&Edit")
        menu.addSeparator()

        addBlockAction = menu.addAction('Add &Patch')
        addBlockAction.setVisible(True)
        addBlockAction.setShortcuts({'A'})
        addBlockAction.triggered.connect(self.add_patch_default)

        # View Menu
        menu = self.menuBar().addMenu("&View")

        # Zoom
        def set_zoom():
            val, ret1 = QInputDialog.getInt(self, 'Zoom Level', 'Zoom Level', self.main_widget.zoom)
            if ret1:
                self.main_widget.zoom = val
        self.zoomAction = menu.addAction(f'Zoom {self.main_widget.zoom}').triggered.connect(set_zoom)

        def menuViewUpdate():
            self.zoomAction.setText(f'Zoom {self.main_widget.zoom}')
        menu.aboutToShow.connect(menuViewUpdate)

        # Help message
        menu = self.menuBar().addMenu("&Help")

        def show_help():
            QMessageBox.about(self, "About", PatchWidget.help_msg)
        menu.addAction('&About', show_help)

    def saveAs(self):
        self.save(save_as=True)

    def save(self, save_as: bool = False):
        if self.patches_file is None or save_as:
            fn, _ = QFileDialog.getSaveFileName(self, 'Save Patches',
                                                filter=('JSON Files (*.json);;All files (*.*)'))

            if fn == '':
                return

            self.patches_file = fn

        ob = {}

        for patch in self.main_widget.patches:
            all = []
            for row in patch.blocks:
                for block in row:
                    e = {}
                    e['row'] = block.row
                    e['col'] = block.col
                    e['label'] = block.label
                    all.append(e)
            ob[patch.name] = all

        json_string = json.dumps(ob, separators=(',', ":"))  # Compact JSON structure
        open(fn, "w+", 1).write(json_string)
        print(f'> Exported {len(self.main_widget.patches)} patches to "{fn}"')

    def open(self, fn: str = ''):
        if fn == '':
            fn, _ = QFileDialog.getOpenFileName(self, 'Open patches.json', filter=(f'JSON Files (*.json);;' + 'All files (*.*)'))

        if fn == '':
            return

        self.patches_file = fn

        with open(fn) as f:
            ob = json.load(f)

        print(ob)

        # patches = [Patch(e) for e in ob]

        self.main_widget.sigPatchesChanged.emit()
        self.main_widget.update()

    def close(self):
        btn = QMessageBox.warning(self, 'Saving', 'Save under recent config?',
                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)

        if btn == QMessageBox.StandardButton.Yes:
            base_path = os.path.join('', '_configs')
            now = datetime.now()
            format = now.strftime(f'%m-%d-%Y_%H-%M-%S')
            fn = os.path.join(base_path, f'{format}.json')
            if not os.path.isdir(base_path):
                os.mkdir(base_path)
            self.recent_configs.append(fn)
            self.save_config(fn)

        print("Closing...")
        super().close()


if __name__ == "__main__":
    app = QApplication([])
    window = Window()

    window.resize(1920, 1080)
    window.show()

    code = app.exec()

    # window.close()
    sys.exit(code)
