from PyQt5.QtGui import QKeySequence, QFont
from PyQt5.QtCore import (Qt, pyqtSignal, pyqtSlot)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStatusBar, QDockWidget, QFileDialog,
    QMessageBox, QInputDialog, QLineEdit, QGridLayout, QPushButton, QMenu, QAbstractItemView, QSlider, QHBoxLayout
)

import pyqtgraph as pg

from copy import deepcopy
import sys
import json

class Block():
    def __init__(self, label: str = '**'):
        self.label = label

    def on_label_changed(self, text: str):
        self.label = text.upper()
        print(f'> Edited block to "{self.label}"')
    def __str__(self):
        return self.label


class Patch():
    def __init__(self, name: str, blocks: list[list[Block]]):
        self.name = name
        self.blocks = blocks

    @classmethod
    def new(self, name: str = 'NewPatch', rows: int = 3, cols: int = 3):
        return self(name, [[Block() for y in range(cols)] for x in range(rows)])

    def duplicate(self):
        return Patch(self.name, deepcopy(self.blocks))

    def rows(self):
        return len(self.blocks)

    def cols(self):
        return len(self.blocks[0]) if self.rows() > 1 else 0

    def setCols(self, val: int, asdelta: bool = False):
        if not asdelta:
            val = val - self.cols()
        if val > 0:
            for i in range(val):
                col = self.cols()-1
                for row in range(self.rows()):
                    self.blocks[row].insert(col+1, Block())
        elif val < 0:
            if self.cols() > 2:
                for i in range(abs(val)):
                    for row in range(self.rows()):
                        self.blocks[row].pop(self.cols()-1)

    def setRows(self, val: int, asdelta: bool = False):
        if not asdelta:
            val = val - self.rows()
        if val > 0:
            for i in range(val):
                col = [Block() for col in range(self.cols())]
                self.blocks.append(col)
        elif val < 0:
            if self.rows() > 2:
                for i in range(abs(val)):
                    self.blocks.pop(self.rows()-1)


class CustomLineEdit(QLineEdit):
    '''Selects the text upon being clicked'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mousePressEvent(self, a0):
        super().mousePressEvent(a0)
        self.selectAll()


class PatchWidget(QWidget):
    sigPatchesChanged = pyqtSignal(name='patchesChanged')
    sigPatchSelected = pyqtSignal(int, name='patchSelected')
    sigUpdateTable = pyqtSignal(name='updateTable')

    help_msg = '''
    1. Add patches to the list by pressing "CTRL+E" or choosing "Add" fromm the context menu
    2. Name the patch and set the shape (rows, columns)
    3. Enter the labels for each block
    4. Move patches up or down the list using "Shift+Up" or "Shift+Down"
    5. Duplicate patches up or down using "Shift+Alt+Up" or "Shift+Alt+Down"
    6. Use "File -> Load/Save" to load or save a patches.json file
    '''

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.font_name = 'Consolas'
        self.title_font_size = 32
        self.plus_minus_buttons_font_size = 24
        self.plus_minus_buttons_thickness = 45
        self.labels_font_size = 36
        self.ui_scale = 1.0

        self.patches: list[Patch] = []
        self.selectedPatch: Patch = None

        self.sigPatchSelected.connect(self.on_patch_selected)

        layout = QGridLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.setLayout(layout)

        self.update()

    def clear_layout(self):
        while self.layout().count() > 0:
            item = self.layout().itemAt(0)
            widget = item.widget()
            if widget is None:
                self.layout().removeItem(item)
            else:
                self.layout().removeWidget(widget)

    def title_changed(self, name: str):
        if self.selectedPatch:
            self.selectedPatch.name = name
            self.sigUpdateTable.emit()
            print(f'> Renamed patch to "{name}"')

    def add_row(self):
        self.selectedPatch.setRows(1, True)
        self.sigPatchesChanged.emit()
        print(f'> Added row to "{self.selectedPatch.name}" - New shape: [{self.selectedPatch.rows()}]x[{self.selectedPatch.cols()}]')

    def rem_row(self):
        if (self.selectedPatch.rows() > 1):
            self.selectedPatch.setRows(-1, True)
            self.sigPatchesChanged.emit()

    def add_col(self):
        self.selectedPatch.setCols(1, True)
        self.sigPatchesChanged.emit()

    def rem_col(self):
        if (self.selectedPatch.cols() > 1):
            self.selectedPatch.setCols(-1, True)
            self.sigPatchesChanged.emit()

    def update(self):
        self.clear_layout()

        if self.selectedPatch:
            title = QLineEdit()
            title.setText(self.selectedPatch.name)
            title.setFont(QFont(self.font_name, int(self.ui_scale*self.title_font_size)))
            title.textChanged[str].connect(self.title_changed)
            title.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            self.layout().addWidget(title, 0, 0, 1, self.selectedPatch.cols())

            for x, col in enumerate(self.selectedPatch.blocks):
                for y, block in enumerate(col):
                    self.layout().addWidget(self.getLineWidget(block), x+1, y)

            add_col_btn = QPushButton()
            add_col_btn.clicked.connect(self.add_col)
            add_col_btn.setText('+')
            add_col_btn.setFont(QFont(self.font_name, int(self.ui_scale*self.plus_minus_buttons_font_size)))
            add_col_btn.setFixedWidth(int(self.ui_scale*self.plus_minus_buttons_thickness))
            add_col_btn.setSizePolicy(1, 3)
            self.layout().addWidget(add_col_btn, 1, self.selectedPatch.cols(), self.selectedPatch.rows(), 1)
            rem_col_btn = QPushButton()
            rem_col_btn.clicked.connect(self.rem_col)
            rem_col_btn.setText('-')
            rem_col_btn.setFont(QFont(self.font_name, int(self.ui_scale*self.plus_minus_buttons_font_size)))
            rem_col_btn.setFixedWidth(int(self.ui_scale*self.plus_minus_buttons_thickness))
            rem_col_btn.setSizePolicy(1, 3)
            self.layout().addWidget(rem_col_btn, 1, self.selectedPatch.cols()+1, self.selectedPatch.rows(), 1)

            add_row_btn = QPushButton()
            add_row_btn.clicked.connect(self.add_row)
            add_row_btn.setText('+')
            add_row_btn.setFont(QFont(self.font_name, int(self.ui_scale*self.plus_minus_buttons_font_size)))
            add_row_btn.setFixedHeight(int(self.ui_scale*self.plus_minus_buttons_thickness))
            self.layout().addWidget(add_row_btn, self.selectedPatch.rows()+1, 0, 1, self.selectedPatch.cols())
            rem_row_btn = QPushButton()
            rem_row_btn.clicked.connect(self.rem_row)
            rem_row_btn.setText('-')
            rem_row_btn.setFont(QFont(self.font_name, int(self.ui_scale*self.plus_minus_buttons_font_size)))
            rem_row_btn.setFixedHeight(int(self.ui_scale*self.plus_minus_buttons_thickness))
            self.layout().addWidget(rem_row_btn, self.selectedPatch.rows()+2, 0, 1, self.selectedPatch.cols())
        else:
            title = QPushButton()
            title.setText('...' if len(self.patches) > 0 else 'Add patches by pressing\n"CTRL+E"')
            title.setFont(QFont(self.font_name, int(self.ui_scale*self.title_font_size)))
            title.setEnabled(False)
            self.layout().addWidget(title, 0, 0, 1, 5)

    @pyqtSlot(int)
    def on_patch_selected(self, index: int):
        if index == -1:
            self.selectedPatch = None
            print(f'> Deselect')
            self.update()
        else:
            patch = self.patches[index]
            if self.selectedPatch != patch:
                self.selectedPatch = patch
                # print(f'> Selected ({index}) "{patch.name}" - Shape: [{patch.rows()}]x[{patch.cols()}]')
                self.update()

    def getLineWidget(self, block: Block):
        line = CustomLineEdit(None)
        line.setText(block.label)
        line.textChanged[str].connect(block.on_label_changed)

        # line.editingFinished.connect(line.clearFocus)
        line.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        line.setMaxLength(2)
        line.setFixedWidth(int(100*self.ui_scale))

        f = QFont(self.font_name, int(self.labels_font_size*self.ui_scale))
        f.setCapitalization(True)
        f.setKerning(True)
        f.setBold(True)
        line.setFont(f)

        return line

    def get_unique_patch_name(self, desired_name: str):
        # names have to be unique
        unique_name = desired_name
        count = 1
        while unique_name in [p.name for p in self.patches]:
            unique_name = f'{desired_name}{count}'
            count += 1
        return unique_name


class Window(QMainWindow):
    def __init__(self):
        super().__init__(parent=None)
        self.setWindowTitle("Patches Tool")

        self.main_widget = PatchWidget()
        self.setCentralWidget(self.main_widget)

        def update_table_only():
            names_col = [self.table.model().data(self.table.indexFromItem(self.table.item(i,0))) for i in range(self.table.rowCount())]
            rows_col = [self.table.model().data(self.table.indexFromItem(self.table.item(i,1))) for i in range(self.table.rowCount())]
            cols_col = [self.table.model().data(self.table.indexFromItem(self.table.item(i,2))) for i in range(self.table.rowCount())]
            
            curr_list = [[names_col[i], rows_col[i], cols_col[i]] for i in range(self.table.rowCount())]
            new_list = [[p.name, p.rows(), p.cols()] for p in self.main_widget.patches]

            if curr_list != new_list:
                self.table.setData(new_list)
                self.table.setHorizontalHeaderLabels(["Name", "Rows", "Cols"])
                if self.table.columnWidth(0) < 200:
                    self.table.setColumnWidth(0, 200)
                self.table.setColumnWidth(1, 100)
                self.table.setColumnWidth(2, 100)
            else:
                print('\nSAME\n')

        def update_all():
            self.unsaved_changes = True
            update_table_only()
            self.main_widget.update()

        self.main_widget.sigPatchesChanged.connect(update_all)
        self.main_widget.sigUpdateTable.connect(update_table_only)

        # Patches List
        self._createPatchesViews()

        # Menu
        self._createMenu()

        # Status bar
        self.status = QStatusBar()
        self.status.showMessage('Init')
        self.setStatusBar(self.status)

        self.patches_file = None
        self.ui_scale = 1.0
        self.unsaved_changes = False

    #############
    # ADD PATCH #
    #############

    def add_default_patch(self):
        self.add_patch()

    def add_patch(self, patch: Patch = None, insert_index: int = -1):

        if patch == None:
            patch = Patch.new(rows=3, cols=3)

        patch.name = self.main_widget.get_unique_patch_name(patch.name)
        self.main_widget.patches.append(patch) if insert_index == -1 else self.main_widget.patches.insert(insert_index, patch)

        print(f'> Added patch "{patch.name}" - Shape: [{patch.rows()}]x[{patch.cols()}]')

        self.main_widget.sigPatchesChanged.emit()

        self.table.selectRow(self.table.rowCount()-1) if insert_index == -1 else self.table.selectRow(insert_index+1)

    ################
    # REMOVE PATCH #
    ################

    def remove_patch(self):
        idx = self.table.currentIndex().row()

        self.main_widget.patches.pop(idx)
        self.main_widget.selectedPatch = None

        self.main_widget.sigPatchesChanged.emit()

    def remove_all(self):
        self.main_widget.patches.clear()
        self.main_widget.selectedPatch = None

        self.main_widget.sigPatchesChanged.emit()

    ###################
    # DUPLICATE PATCH #
    ###################

    def duplicate_up(self, down: bool = False):
        idx = self.table.currentIndex().row()

        self.add_patch(self.main_widget.patches[idx].duplicate(), idx+1 if down else idx)
        print(f'> Duplicated "{self.main_widget.patches[idx].name}" {"down" if down else "up"}')
        self.table.selectRow(idx+1 if down else idx)

    def duplicate_down(self):
        self.duplicate_up(True)

    ##############
    # MOVE PATCH #
    ##############

    def move_up(self, down: bool = False):
        idx = self.table.currentIndex().row()

        if down:
            dir = 1
            cond = idx < len(self.main_widget.patches)-1 and idx >= 0
        else:
            dir = -1
            cond = idx < len(self.main_widget.patches) and idx > 0

        if cond:
            self.main_widget.patches[idx], self.main_widget.patches[idx + dir] = self.main_widget.patches[idx + dir], self.main_widget.patches[idx]
            print(f'> Swapped [{idx}] <=> [{idx + dir}]')

        self.main_widget.selectedPatch = self.main_widget.patches[idx+dir]

        self.main_widget.sigPatchesChanged.emit()
        self.table.selectRow(idx+dir)

    def move_down(self):
        self.move_up(True)

    ##################
    # TABLE BINDINGS #
    ##################

    def row_selected(self):
        idx = self.table.currentIndex().row()

        self.main_widget.sigPatchSelected.emit(idx)

    def row_deselect_all(self):
        self.table.clearSelection()
        self.main_widget.sigPatchSelected.emit(-1)

    def row_changed(self):
        if len(self.table.selectedIndexes()) == 0:
            return
        idx = self.table.currentIndex().row()

        new_name = self.table.item(idx, 0).value
        new_rows = max(2, self.table.item(idx, 1).value)
        new_cols = max(2, self.table.item(idx, 2).value)

        if new_name != self.main_widget.patches[idx].name:
            self.main_widget.patches[idx].name = self.main_widget.get_unique_patch_name(new_name).strip().replace(' ', '_')

        self.main_widget.patches[idx].setRows(new_rows)
        self.main_widget.patches[idx].setCols(new_cols)

        self.main_widget.sigPatchesChanged.emit()
        self.table.selectRow(idx)

    ######
    # UI #
    ######

    def _createPatchesViews(self):
        # List of patches
        self.table = pg.TableWidget(editable=True, sortable=True)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setFont(QFont('Consolas', 18))
        self.table.itemSelectionChanged.connect(self.row_selected)
        self.table.model().dataChanged.connect(self.row_changed)
        self.table.setSortingEnabled(False)

        menu: QMenu = self.table.contextMenu
        menu.addSeparator()

        menu.addAction('Add').triggered.connect(self.add_default_patch)
        menu.addAction('Duplicate').triggered.connect(self.duplicate_down)
        rem_action = menu.addAction('Remove')
        rem_action.triggered.connect(self.remove_patch)
        rem_action.setShortcuts([QKeySequence.StandardKey.Delete])

        table_widget = QDockWidget('List of Patches')
        table_widget.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        table_widget.setMinimumWidth(600)
        table_widget.setWidget(self.table)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, table_widget)

    def _createMenu(self):
        # File menu
        menu = self.menuBar().addMenu("&File")
        menu.addAction("&Open...", self.open, shortcut='Ctrl+O')
        menu.addSeparator()
        menu.addAction("&Import...", self.import_patches, shortcut='Ctrl+I')
        menu.addSeparator()
        menu.addAction("&Save...", self.save, shortcut='Ctrl+S')
        menu.addAction("Save As...", self.saveAs)
        menu.addSeparator()
        menu.addAction("&Exit", self.close)

        # Edit menu
        menu = self.menuBar().addMenu("&Edit")
        menu.addSeparator()

        deselectAction = menu.addAction('De&select')
        deselectAction.setVisible(True)
        deselectAction.setShortcuts(['ESC'])
        deselectAction.triggered.connect(self.row_deselect_all)

        menu.addSeparator()

        addBlockAction = menu.addAction('&Add Patch')
        addBlockAction.setVisible(True)
        addBlockAction.setShortcuts({'CTRL+E', })
        addBlockAction.triggered.connect(self.add_default_patch)

        addBlockAction = menu.addAction('Duplicate Patch Up')
        addBlockAction.setVisible(True)
        addBlockAction.setShortcuts({'Shift+Alt+Up'})
        addBlockAction.triggered.connect(self.duplicate_up)

        addBlockAction = menu.addAction('Duplicate Patch Down')
        addBlockAction.setVisible(True)
        addBlockAction.setShortcuts({'Shift+Alt+Down'})
        addBlockAction.triggered.connect(self.duplicate_down)

        removeBlockAction = menu.addAction('&Remove Current Patch')
        removeBlockAction.setVisible(True)
        removeBlockAction.setShortcuts([QKeySequence.StandardKey.Delete])
        removeBlockAction.triggered.connect(self.remove_patch)

        menu.addSeparator()

        moveUpAction = menu.addAction('Move Up')
        moveUpAction.setVisible(True)
        moveUpAction.setShortcuts({'Alt+Up'})
        moveUpAction.triggered.connect(self.move_up)

        moveDownAction = menu.addAction('Move Down')
        moveDownAction.setVisible(True)
        moveDownAction.setShortcuts({'Alt+Down'})
        moveDownAction.triggered.connect(self.move_down)

        menu.addSeparator()

        removeBlockAction = menu.addAction('Remove All Patches')
        removeBlockAction.setVisible(True)
        removeBlockAction.triggered.connect(self.remove_all)

        # View Menu
        menu = self.menuBar().addMenu("&View")

        # UI scale
        def set_ui_scale():
            val, ret1 = QInputDialog.getDouble(self, 'UI Scale', 'UI Scale', self.ui_scale, min=0.0)
            if ret1:
                self.ui_scale = val
                self.main_widget.ui_scale = val
                self.main_widget.update()
        self.ui_scale_action = menu.addAction('')
        self.ui_scale_action.triggered.connect(set_ui_scale)

        def menuViewUpdate():
            self.ui_scale_action.setText(f'UI Scale: {self.ui_scale}')
        menu.aboutToShow.connect(menuViewUpdate)

        # Help message
        menu = self.menuBar().addMenu("&Help")

        def show_help():
            QMessageBox.about(self, "About", PatchWidget.help_msg)
        menu.addAction('&About', show_help)

    #######
    # OPS #
    #######

    def saveAs(self):
        self.save(save_as=True)

    def save(self, save_as: bool = False):
        if self.patches_file is None or save_as:
            fn, _ = QFileDialog.getSaveFileName(self, 'Save Patches',
                                                filter=('JSON Files (*.json);;All files (*.*)'))

            if fn == '':
                return False

            self.patches_file = fn

        ob = {}

        for patch in self.main_widget.patches:
            all :list[str]= []
            for _, row in enumerate(patch.blocks):
                block_strings = []
                for _, block in enumerate(row):
                    block_strings.append(block.label.upper())
                all.append(block_strings)
            ob[patch.name] = all
        
        json_string = json.dumps(ob, separators=(',', ":"))  # Compact JSON structure
        open(self.patches_file, "w+", 1).write(json_string)
        print(f'> Saved {len(self.main_widget.patches)} patches to "{self.patches_file}"')

        self.status.showMessage(f'> Saved to file: {self.patches_file}')
        self.unsaved_changes = False

        return True

    def import_patches(self):
        self.open(fn='', do_import=True)

    def open(self, fn: str = '', do_import:bool = False):
        if fn == '':
            if len(self.main_widget.patches) > 0 and not do_import:
                btn = QMessageBox.warning(self, 'Warning', 'You will lose unsaved patches by loading from a patches.json file.\nContinue?',
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)

                if btn == QMessageBox.StandardButton.No:
                    return

            fn, _ = QFileDialog.getOpenFileName(self, f'{"Import" if do_import else "Open"} patches.json', filter=(f'JSON Files (*.json);;' + 'All files (*.*)'))

        if fn == '':
            return

        if not do_import:
            self.patches_file = fn

        with open(fn) as f:
            ob = json.load(f)

        if not do_import:
            self.main_widget.patches.clear()
            self.main_widget.sigPatchSelected.emit(-1)

        for name in ob:            
            patch = Patch.new(self.main_widget.get_unique_patch_name(name), len(ob[name]), len(ob[name][0]))
            for row_idx in range(len(ob[name])):
                for col_idx in range(len(ob[name][row_idx])):
                    patch.blocks[row_idx][col_idx].label = ob[name][row_idx][col_idx].strip().replace(' ', '_')

            self.main_widget.patches.append(patch)

        self.main_widget.sigPatchesChanged.emit()
        self.unsaved_changes =  do_import
        self.status.showMessage(f'> {"Imported" if do_import else "Opened"} file: {fn}')

    def close(self):

        while self.patches_file == None or self.unsaved_changes:
            btn = QMessageBox.warning(self, 'Save Changes', 'Do you wish to save the changes before exiting?',
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)

            if btn == QMessageBox.StandardButton.No:
                break

            if self.patches_file:
                btn = QMessageBox.warning(self, 'Override', f'Do you wish to override the file?\n{self.patches_file}',
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)

                self.saveAs() if btn == QMessageBox.StandardButton.No else self.save()
                
            self.saveAs()

        print("> Bye Bye!")
        super().close()


if __name__ == "__main__":
    app = QApplication([])
    window = Window()

    window.resize(1920, 1080)
    window.show()

    code = app.exec()

    window.close()
    sys.exit(code)
