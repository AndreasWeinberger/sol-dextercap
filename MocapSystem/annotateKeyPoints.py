
import scipy.spatial
from . import utils
# from ..VideoProcess.test_conv3 import CornerLabeler
from VideoProcess.test_conv import CornerLabeler

from PyQt5.QtGui import QIntValidator, QMouseEvent, QImage, QPixmap, QWheelEvent, QKeySequence
from PyQt5.QtCore import (Qt, QSize, QTimer, QEvent, pyqtSignal, pyqtSlot)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStatusBar, QDockWidget, QDialog,
    QToolBar, QFileDialog, QMessageBox, QInputDialog,
    QLineEdit, QSlider,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QLayout, QPushButton, QSizePolicy, QLabel,
    QMenu, QAction, QFrame
)

import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw

import pyqtgraph as pg
import qimage2ndarray

import numpy as np
import scipy
import cv2

import os
import sys
import pathlib
import json
import platform
from functools import partial

from typing import Tuple, Dict, List, Any, Union

from datetime import datetime


class DataLoader:
    def __init__(self, fn: List[str]) -> None:
        self.fn = fn
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    def image_size(self) -> Tuple[int, int]:
        # return: (width, height)
        raise NotImplementedError


class MultipleFileLoader(DataLoader):
    def __init__(self, fn: List[str], sub_loaders: List[DataLoader]) -> None:
        super().__init__(fn)
        self.sub_loaders = sub_loaders

        self.sub_loader_frames = [len(loader) for loader in sub_loaders]

        self.num_frames = sum(self.sub_loader_frames)
        self.fps = 1
        self.width, self.height = sub_loaders[0].image_size()
        for loader in sub_loaders[1:]:
            assert (self.width, self.height) == loader.image_size()

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.num_frames:
            raise IndexError

        for num_frame, loader in zip(self.sub_loader_frames, self.sub_loaders):
            if idx >= num_frame:
                idx -= num_frame
                continue

            return loader[idx]

        raise IndexError

    def image_size(self) -> Tuple[int, int]:
        # return: (width, height)
        return (self.width, self.height)


class ImageLoader(DataLoader):
    def __init__(self, fn: str) -> None:
        super().__init__(fn)

        self.image: np.ndarray = cv2.imread(self.fn[0], flags=cv2.IMREAD_COLOR)
        self.image = self.image.transpose((1, 0, 2))
        self.image = self.image[:, :, ::-1]  # bgr -> rgb

    def __len__(self) -> int:
        if self.image is not None:
            return 1
        else:
            return 0

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.image

    def image_size(self) -> Tuple[int, int]:
        # return: (width, height)
        if self.image is None:
            return (0, 0)

        return (self.image.shape[1], self.image.shape[0])


class VideoLoader(DataLoader):
    def __init__(self, fn: str, buffer_size: int = 100) -> None:
        super().__init__([fn])

        self.video_cap = cv2.VideoCapture(self.fn[0])
        self.curr_frame = 0

        self.num_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __del__(self):
        if self.video_cap is not None:
            self.video_cap.release()

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.num_frames:
            raise IndexError

        if idx != self.curr_frame + 1:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = self.video_cap.read()
        frame = frame.transpose((1, 0, 2))
        frame = frame[:, :, ::-1]  # bgr -> rgb
        self.curr_frame = idx
        return frame

    def image_size(self) -> Tuple[int, int]:
        # return: (width, height)
        return (self.width, self.height)


class RawVideoLoader(DataLoader):
    def __init__(self, fn: str, width: int | None = None, height: int | None = None, fps: int | None = None) -> None:
        super().__init__([fn])

        with open(self.fn[0], 'br') as f:
            self.frames = np.fromfile(f, dtype=np.uint8)

        if width is None:
            name = os.path.splitext(fn)[0]
            for token in name.split('_'):
                if len(token) < 2:
                    continue

                if token[0] == 'w':
                    width = int(token[1:])
                if token[0] == 'h':
                    height = int(token[1:])
                if token[0] == 'f':
                    fps = int(token[1:])

        self.width = width
        self.height = height
        self.fps = fps

        self.frames = self.frames.reshape(-1, self.height, self.width)
        self.num_frames = self.frames.shape[0]

        self.curr_frame = 0

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.num_frames:
            raise IndexError

        frame = self.frames[idx, ::-1]

        frame = frame.reshape(self.height, self.width, 1)
        frame = np.broadcast_to(frame, (self.height, self.width, 3))

        return frame

    def image_size(self) -> Tuple[int, int]:
        # return: (width, height)
        return (self.width, self.height)


class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        btnLayout = self.createButtons()
        layout.addLayout(btnLayout)

        self.frameEdit = QLineEdit()
        self.frameEdit.setMaximumWidth(100)
        self.editValidator = QIntValidator()
        self.frameEdit.setValidator(self.editValidator)
        self.frameSlider = QSlider(Qt.Horizontal)

        def updateEdit(sliderVal):
            self.frameEdit.setText(str(sliderVal))

        def updateSlider(editVal):
            if editVal == '':
                self.frameSlider.setValue(0)
            else:
                self.frameSlider.setValue(int(editVal))

        self.frameSlider.valueChanged.connect(updateEdit)
        self.frameEdit.textEdited.connect(updateSlider)

        layout.addSpacing(5)
        layout.addWidget(self.frameSlider)
        layout.addSpacing(7)
        layout.addWidget(self.frameEdit)

        layout.setSpacing(3)

        self.setRange(0, 100)
        self.frameSlider.setValue(0)
        self.frameEdit.setText('0')

    def createButtons(self):
        self.stepBackBtn = QPushButton('<')
        self.stepBackBtn.setFixedSize(40, 40)
        self.stepForwardBtn = QPushButton('>')
        self.stepForwardBtn.setFixedSize(40, 40)
        self.pauseBtn = QPushButton('|| PAUSE ||')
        self.pauseBtn.setFixedSize(203, 40)
        self.playBtn = QPushButton('PLAY >>')
        self.playBtn.setFixedSize(100, 40)
        self.backwardPlayBtn = QPushButton('<< REV')
        self.backwardPlayBtn.setFixedSize(100, 40)

        self.stepBackBtn.clicked.connect(lambda: self.step(-1))
        self.stepForwardBtn.clicked.connect(lambda: self.step(1))
        self.playBtn.clicked.connect(lambda: self.play(1))
        self.backwardPlayBtn.clicked.connect(lambda: self.play(-1))
        self.pauseBtn.clicked.connect(self.pause)

        layout = QHBoxLayout()
        layout.setSpacing(3)
        layout.setContentsMargins(0, 0, 0, 0)
        for btn in [self.stepBackBtn,
                    self.backwardPlayBtn,
                    self.pauseBtn,
                    self.playBtn,
                    self.stepForwardBtn,
                    ]:
            layout.addWidget(btn)
        self.pauseBtn.hide()

        self.playOffset = 0
        self.playTimer = QTimer()
        self.playTimer.timeout.connect(lambda: self.step(self.playOffset))

        return layout

    def step(self, offset):
        v = self.frameSlider.value() + offset
        if v > self.frameSlider.maximum():
            v = self.frameSlider.minimum()

        if v < self.frameSlider.minimum():
            v = self.frameSlider.maximum()

        self.frameSlider.setValue(v)

    def play(self, offset):
        self.pauseBtn.show()
        self.playBtn.hide()
        self.backwardPlayBtn.hide()

        self.playOffset = offset
        self.playTimer.start(16)

    def pause(self):
        self.pauseBtn.hide()
        self.playBtn.show()
        self.backwardPlayBtn.show()

        self.playTimer.stop()

    def setRange(self, lower, upper):
        self.frameSlider.setRange(lower, upper - 1)
        self.editValidator.setRange(lower, max(lower, upper - 1))

    def setValue(self, v):
        if v > self.frameSlider.maximum():
            v = self.frameSlider.minimum()

        if v < self.frameSlider.minimum():
            v = self.frameSlider.maximum()

        self.frameSlider.setValue(v)


class ImageItem(pg.ImageItem):
    sigMouseClickedImage = pyqtSignal(float, float, Qt.KeyboardModifiers, name='imageClicked')
    sigMouseDoubleClickedImage = pyqtSignal(float, float, Qt.KeyboardModifiers, name='imageDoubleClicked')
    sigMouseDraggedImage = pyqtSignal(float, float, Qt.KeyboardModifiers, name='imageDragged')

    def __init__(self, image=None, **kargs):
        super().__init__(image, **kargs)

        self.setBorder({'color': "#FF0", 'width': 2})

    def mouseDragEvent(self, event):
        super().mouseDragEvent(event)

        if event.isAccepted():
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.sigMouseDraggedImage.emit(event.pos().x(), event.pos().y(), event.modifiers())

    def mouseClickEvent(self, event):
        super().mouseClickEvent(event)

        if event.isAccepted():
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.sigMouseClickedImage.emit(event.pos().x(), event.pos().y(), event.modifiers())
            # QMessageBox.information(None, 'Clicked!', f'({event.pos().x(), event.pos().y()})')

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)

        if event.isAccepted():
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.sigMouseDoubleClickedImage.emit(event.pos().x(), event.pos().y(), event.modifiers())
            # QMessageBox.information(None, 'Double Clicked!', f'({event.pos().x(), event.pos().y()})')


class BlockAnnotatorWidget(QDialog):
    def __init__(self, parent: QWidget | None, image: np.ndarray, label: str, direction: int,
                 label_candidates: List[List[str]] | None = None, image_scale_log: int = 0,
                 used_labels: List[str] | None = None
                 ) -> None:
        super().__init__(parent)

        self.image = image
        if image is not None:
            self.image = np.asarray(image)
            if self.image.ndim == 2:
                self.image = self.image[..., None]
            if self.image.shape[-1] == 1:
                self.image = np.concatenate([self.image, self.image, self.image], axis=-1)

            if np.issubdtype(self.image.dtype, np.floating):
                self.image = np.round(self.image * 255, 0)
            self.image = self.image.astype(np.uint8)

        self.direction = direction
        self.label = label

        self.image_scale_log = image_scale_log
        self.label_candidates = label_candidates
        self.used_labels = used_labels

        self.setWindowTitle('Block Annotator')
        self._setupUI()

        self.showImage()
        self.labelEditor.setFocus()

    def _setupUI(self):
        layout0 = QVBoxLayout()
        layout = QHBoxLayout()

        self.imageLabel = QLabel()
        self.imageLabel.setFixedSize(400, 400)
        self.imageLabel.setStyleSheet('background: green;')
        self.imageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.imageLabel.setMinimumWidth(100)
        layout.addWidget(self.imageLabel)

        button = QPushButton('\u21BA')
        button.clicked.connect(self.rotateImage)
        button.setStyleSheet('font-size: 80pt; font-weight:bold')
        button.setFixedSize(400, 400)
        button.setShortcut('Ctrl+R')
        layout.addWidget(button)
        layout0.addLayout(layout)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #c0c0c0;")
        layout0.addWidget(line)

        layout = QHBoxLayout()
        label = QLabel('Label: ')
        label.setFixedHeight(50)
        label.setStyleSheet('font-size: 20pt;')
        layout.addWidget(label)
        self.labelEditor = QLineEdit(self.label)
        self.labelEditor.setStyleSheet('font-size: 20pt;')
        self.labelEditor.setFixedHeight(50)

        def label_text_changed(text):
            self.label = text
        self.labelEditor.textChanged.connect(label_text_changed)
        layout.addWidget(self.labelEditor)
        layout0.addLayout(layout)

        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        button = QPushButton('OK')
        button.setStyleSheet('font-size: 20pt;')
        button.setFixedWidth(200)
        button.clicked.connect(self.accept)
        button.setDefault(True)
        button.setEnabled(False)
        self.labelEditor.textChanged.connect(partial(button.setEnabled, True))
        layout.addWidget(button)

        button = QPushButton('Cancel')
        button.setStyleSheet('font-size: 20pt;')
        button.setFixedWidth(200)
        button.clicked.connect(self.reject)
        layout.addWidget(button)

        layout0.addLayout(layout)

        if self.label_candidates is not None:
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            line.setStyleSheet("background-color: #c0c0c0;")
            layout0.addWidget(line)

            layoutB = QHBoxLayout()
            layoutB.setAlignment(Qt.AlignmentFlag.AlignLeft)
            layout0.addLayout(layoutB)
            layout = QGridLayout()
            layoutB.addLayout(layout)

            flagged_button = None

            def btn_clicked(btn, text):
                nonlocal flagged_button
                self.labelEditor.setText(text)
                if flagged_button is not None:
                    if self.used_labels is not None and flagged_button.text() in self.used_labels:
                        flagged_button.setStyleSheet("background-color: aqua;")
                    else:
                        flagged_button.setStyleSheet(None)
                btn.setStyleSheet("background-color: lime;")
                flagged_button = btn

            for row, label_row in enumerate(self.label_candidates):
                if len(label_row) == 0:
                    layout.setRowMinimumHeight(row, 30)
                    blk = QLabel()
                    blk.setStyleSheet("background-color: #c0c0c0;")
                    layout.addWidget(blk, row, 0, 1, -1)
                    continue

                for col, label_text in enumerate(label_row):
                    if label_text == ':':
                        blk = QLabel()
                        blk.setStyleSheet("background-color: #c0c0c0;")
                        blk.setFixedWidth(50)
                        blk.setFixedHeight(50)
                        layout.addWidget(blk, row, col)
                        continue

                    if label_text == '':
                        layout.setColumnMinimumWidth(col, 50)
                        continue

                    button = QPushButton(label_text)
                    button.setFixedWidth(50)
                    button.setFixedHeight(50)
                    if label_text == self.label:
                        flagged_button = button
                        button.setStyleSheet("background-color: lime;")

                    elif self.used_labels is not None and label_text in self.used_labels:
                        button.setStyleSheet("background-color: aqua;")

                    button.clicked.connect(partial(btn_clicked, button, label_text))
                    layout.addWidget(button, row, col)

            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            line.setStyleSheet("background-color: #c0c0c0;")
            layout0.addWidget(line)

            layout = QHBoxLayout()
            layout.setAlignment(Qt.AlignmentFlag.AlignRight)
            button = QPushButton('OK')
            button.setStyleSheet('font-size: 20pt;')
            button.setFixedWidth(200)
            button.setEnabled(False)
            self.labelEditor.textChanged.connect(partial(button.setEnabled, True))
            button.clicked.connect(self.accept)
            layout.addWidget(button)

            button = QPushButton('Cancel')
            button.setStyleSheet('font-size: 20pt;')
            button.setFixedWidth(200)
            button.clicked.connect(self.reject)
            layout.addWidget(button)

            layout0.addLayout(layout)

        self.setLayout(layout0)

    def rotateImage(self):
        if self.image is None:
            return

        self.direction = (self.direction + 1) % 4
        self.showImage()
        self.labelEditor.setFocus()

    def showImage(self):
        # { 'image': array, 'labels': [], 'direction': int }
        image = self.image
        if self.direction > 0:
            options = [0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
            image = cv2.rotate(image, options[self.direction])

        image = image.transpose((1, 0, 2))

        q_image: QImage = qimage2ndarray.array2qimage(image)

        if self.image_scale_log != 0:
            scale = 10**(self.image_scale_log / 5)

            w = q_image.size().width()
            h = q_image.size().height()
            w = int(w * scale)
            h = int(h * scale)
            q_image = q_image.scaled(w, h)

        # q_image = q_image.scaled(self.imageLabel.size())
        self.imageLabel.setPixmap(QPixmap(q_image))

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        super().wheelEvent(event)

        step = event.angleDelta().y() / 120

        self.image_scale_log += step
        self.image_scale_log = max(-10, self.image_scale_log)
        self.image_scale_log = min(self.image_scale_log, 10)
        # print(self.image_scale_log)

        self.showImage()


class AnnotatorWidget(QWidget):
    sigKeyPointChanged = pyqtSignal(int, dict, name='keypointChanged')

    help_msg = '''
1. Open a video/image file using File -> Open (Ctrl+O)
2. Left-click in the image to place a keypoint, Ctrl+left-click to remove a keypoint
3. Select keypoints in the KeyPointView, then right-click -> remove to remove a keypoint/keypoints
4. File -> Load/Save to interact with an annotation file
5. File -> Export to export keypoints and keypointed images into a folder
    '''

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.dataloader = None
        self.image_item = ImageItem(np.broadcast_to(np.eye(10, dtype=np.uint8)[..., None], (10, 10, 3)))
        self.view = pg.ImageView(self, imageItem=self.image_item)
        self.view.ui.roiBtn.hide()
        self.view.ui.menuBtn.hide()
        # self.view.ui.histogram.hide()

        self.image_item.sigMouseClickedImage.connect(self.on_image_click_event)
        self.image_item.sigMouseDoubleClickedImage.connect(self.on_image_double_click_event)
        #self.image_item.sigMouseDraggedImage.connect(self.on_image_dragged_event)

        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.view)

        self.current_frame_idx = -1
        self.current_frame = np.broadcast_to(np.eye(10, dtype=np.uint8)[..., None], (10, 10, 3))
        self.keypoints_of_frame = {}
        self._selected_keypoints = []
        self._show_keypoints = True
        self._show_blocks = True
        self._show_heatmap = True

        self.video_fn = ''

        self.marker_distance_thr: int | float = 150
        self.marker_confidence_thr: int | float = 0.7
        self.edge_confidence_thr: int | float = 0.6
        self.kp_radius: int | float = 6
        self.kp_thickness: int | float = 1

        self.blk_text_height_store: int | float = 0
        self.blk_text_height: int | float = 1
        self.blk_text_thickness: int | float = 1
        self.blk_text_wrap: bool = True

        self.label_candidates: List[List[str]] | None = None
        self.last_blk_label: str | None = None
        self.last_blk_direction: int | None = None
        self.last_image_scale_log: int = 0

        self.auto_labeler: CornerLabeler = None
        self.keypoint_model: str = ''
        self.edge_model: str = ''

    def open_video(self, file_names: List[str]):
        if len(file_names) == 1:
            fn = file_names[0]
            if fn[-3:] == 'raw':
                self.dataloader = RawVideoLoader([fn])
            else:
                self.dataloader = VideoLoader([fn], buffer_size=-1)
            self.video_fn = fn
            self.reset()

        else:
            loaders = []
            file_names = sorted(file_names, key=lambda fn: os.path.basename(fn))
            for fn in file_names:
                if fn[-3:] == 'raw':
                    loader = RawVideoLoader([fn])
                else:
                    loader = ImageLoader([fn])

                loaders.append(loader)

                print(f'-- loaded {fn}')
                print(f'-- #frames: {len(loader)}')
                print(f'-- image size: {loader.image_size()}')

            self.dataloader = MultipleFileLoader(fn=file_names, sub_loaders=loaders)
            self.video_fn = 'multiple_videos'
            self.reset()

        print(f'loaded {self.video_fn}')
        print(f'#frames: {len(self.dataloader)}')
        print(f'image size: {self.dataloader.image_size()}')

    def open_image(self, fn: str):
        self.dataloader = ImageLoader([fn])
        self.video_fn = fn
        self.reset()

    def save_keypoints(self, fn: str):
        keypoints = [{'frame': idx,
                      'keypoints': self.keypoints_of_frame[idx]['keypoints'],
                      'blocks': self.keypoints_of_frame[idx]['blocks'] if 'blocks' in self.keypoints_of_frame[idx] else []
                      }
                     for idx in self.keypoints_of_frame]

        with open(fn, 'w') as f:
            json.dump(keypoints, f, indent=4)

    def load_keypoints(self, file_names: List[str]):
        if len(file_names) == 0:
            return

        file_names = sorted(file_names, key=lambda fn: os.path.basename(fn))
        self.keypoints_of_frame = {}

        base_frame_indices = [0]
        if isinstance(self.dataloader, MultipleFileLoader):
            base_frame_indices.extend(np.cumsum(self.dataloader.sub_loader_frames).tolist())

        for fn_idx, fn in enumerate(file_names):
            with open(fn, 'r') as f:
                data = json.load(f)

            base_frame_idx = base_frame_indices[min(fn_idx, len(base_frame_indices) - 1)]

            for frame_data in data:
                self.keypoints_of_frame[frame_data['frame'] + base_frame_idx] = {
                    'keypoints': [tuple(pt) for pt in frame_data['keypoints']],
                    'blocks': [{'corners': tuple(tuple(pt) for pt in blk['corners']),
                                'label': blk['label'],
                                'direction': blk['direction'] if 'direction' in blk else 0,
                                } for blk in frame_data['blocks']] if 'blocks' in frame_data else [],
                }

        self._selected_keypoints = []
        self.show_frame(self.current_frame_idx, force_update=True)

    def load_label_candidates(self, fn: str):
        with open(fn, 'r') as f:
            data = json.load(f)

        self.label_candidates = data

    def export(self, folder: str):
        # export keypoints and images to keypoint
        keypoint_fn = os.path.join(folder, 'labels.json')

        base_name = os.path.splitext(os.path.basename(self.video_fn))[0]
        if len(base_name) == 0:
            base_name = 'image'

        def img_fn(idx: int):
            return f'{base_name}_{idx:06d}.jpg'

        # Fix for faulty labels when exporting ?
        # for blk in self.keypoints_of_frame[idx]['blocks']:
        #     if len(blk['label']) > 1:
        #         blk['label'] = str.strip(blk['label']).replace('*','')

        keypoints = [{'frame': idx,
                      'keypoints': self.keypoints_of_frame[idx]['keypoints'],
                      'blocks': [blk for blk in self.keypoints_of_frame[idx]['blocks'] if blk['label'] != ''],
                      'image': {
                          'path': img_fn(idx),
                          'size': [self.dataloader.height, self.dataloader.width],
                          'size-order': 'h,w',
                          'source': pathlib.PurePath(self.video_fn).name,
                      },
                      }
                     for idx in self.keypoints_of_frame]

        with open(keypoint_fn, 'w') as f:
            json.dump(keypoints, f, indent=4)

        for idx in self.keypoints_of_frame:
            fn = os.path.join(folder, img_fn(idx))

            frame = self.dataloader[idx]
            frame = frame[:, :, ::-1]  # rgb -> bgr
            frame = frame.transpose((1, 0, 2))
            cv2.imwrite(fn, frame)

        # save sanity check:
        os.makedirs(os.path.join(folder, '_check'), exist_ok=True)
        for idx in self.keypoints_of_frame:
            fn = os.path.join(folder, '_check', img_fn(idx))

            frame = self.dataloader[idx].copy()
            self._draw_blocks(frame, self.keypoints_of_frame[idx]['blocks'], export_checking=True)
            self._draw_keypoints(frame, self.keypoints_of_frame[idx]['keypoints'], [])
            frame = frame[:, :, ::-1]  # rgb -> bgr
            frame = frame.transpose((1, 0, 2))
            cv2.imwrite(fn, frame)

    def reset(self):
        self.current_frame_idx = -1
        self.current_frame = np.broadcast_to(np.eye(10, dtype=np.uint8)[..., None], (10, 10, 3))
        self.keypoints_of_frame = {}
        self._selected_keypoints = []
        self._show_keypoints = True
        self._show_blocks = True

        self.sigKeyPointChanged.emit(self.current_frame_idx, self.keypoints_of_frame)

    def num_images(self):
        if self.dataloader is None:
            return 0

        return len(self.dataloader)

    @staticmethod
    def _has_close_point(pt: Tuple[float, float], points: List[Tuple[float, float]]):
        # if pt in points:
        #     return True
        for idx, pt1 in enumerate(points):
            if abs(pt[0] - pt1[0]) + abs(pt[1] - pt1[1]) < 1e-4:
                return True, idx, pt1

        return False, -1, None

    def _draw_keypoints(self, image: np.ndarray, keypoints: List[Tuple[float, float]], selected_keypoints: List[Tuple[float, float]]):
        # r = max(1, image.shape[1] // 500)
        # w = max(1, image.shape[1] // 2000)

        r = self.kp_radius
        w = self.kp_thickness

        for pt in keypoints:
            pt_int = tuple(int(x+0.5) for x in pt)
            c = (36, 255, 12)
            if self._has_close_point(pt, selected_keypoints)[0]:
                c = (180, 180, 255)
                continue

            if w > 0:
                cv2.circle(image, (pt_int[1], pt_int[0]), r, c, w, lineType=cv2.LINE_AA)
            cv2.drawMarker(image, (pt_int[1], pt_int[0]), c, cv2.MARKER_CROSS, markerSize=r*2, thickness=1)

        c = (180, 180, 255)
        for pt in selected_keypoints:
            pt_int = tuple(int(x+0.5) for x in pt)
            cv2.drawMarker(image, (pt_int[1], pt_int[0]), c, cv2.MARKER_CROSS, markerSize=r*2, thickness=1)

    def _draw_blocks(self, image: np.ndarray, blocks: List[Dict[str, Any]], export_checking: bool = False):
        h = self.blk_text_height
        thickness = self.blk_text_thickness

        for blk in blocks:
            corners = np.asarray(blk['corners'])
            label = blk['label']
            direction = blk['direction']
            pt = np.round(np.mean(corners, axis=0)).astype(int)

            corners = np.roll(corners, shift=-direction, axis=0)

            center = corners.mean(axis=0, keepdims=True)
            corners_in = (corners - center) * 0.7 + center

            if export_checking and label == '':
                continue

            if not export_checking:
                blk_img_size = 64
                blk_img = (np.ones((blk_img_size, blk_img_size, 3)) * [[230, 250, 50]]).astype(image.dtype)

                # Set up the destination points for the perspective transform
                src = np.array([
                    [0, 0],
                    [blk_img_size - 1, 0],
                    [blk_img_size - 1, blk_img_size - 1],
                    [0, blk_img_size - 1]], dtype=np.float32)

                xy_min = np.floor(corners_in.min(axis=0)).astype(int)
                xy_max = np.ceil(corners_in.max(axis=0)).astype(int)
                blk_size = max(xy_max - xy_min)

                # Calculate the perspective transform matrix and apply it
                M = cv2.getPerspectiveTransform(src, (corners_in - xy_min).astype(np.float32).reshape(-1, 2))
                warped = cv2.warpPerspective(blk_img, M, (blk_size, blk_size), flags=cv2.INTER_CUBIC)
                warped = np.clip(warped, 0, 255)

                warped = warped.swapaxes(0, 1)

                img_blk = image[xy_min[0]:xy_min[0]+warped.shape[0],
                                xy_min[1]:xy_min[1]+warped.shape[1]]

                blk_mask = warped > 0

                alpha = 0.8 if label == '' else 0.9
                img_blk[blk_mask] = img_blk[blk_mask]*alpha + warped[blk_mask]*(1 - alpha)

                for i, pt_i in enumerate(corners_in):
                    clr = [50, 50, 50]
                    if i < 3:
                        clr[i] = 255
                    cv2.circle(image, (int(pt_i[1]), int(pt_i[0])), 1, clr, 1, lineType=cv2.LINE_AA)

            if self.blk_text_wrap:
                # put text using opencv
                c = (36, 255, 12)
                text_img_size, _ = cv2.getTextSize('MW', cv2.FONT_HERSHEY_PLAIN, h, thickness)
                text_img_size = max(text_img_size) + 1
                text_img = np.zeros((text_img_size, text_img_size, 3), dtype=image.dtype)

                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, h, thickness)
                text_w, text_h = text_size

                t_x = max(0, (text_img_size - text_w) // 2)
                t_y = min(max(0, (text_img_size - text_h) // 2 + text_h), text_img_size - 1)
                cv2.putText(text_img, label, (t_x, t_y), cv2.FONT_HERSHEY_PLAIN, h, c, thickness, cv2.LINE_AA)

                # Set up the destination points for the perspective transform
                src = np.array([
                    [0, 0],
                    [text_img_size - 1, 0],
                    [text_img_size - 1, text_img_size - 1],
                    [0, text_img_size - 1]], dtype=np.float32)

                # src = np.roll(src, shift=direction, axis=0)

                xy_min = np.floor(corners.min(axis=0)).astype(int)
                xy_max = np.ceil(corners.max(axis=0)).astype(int)
                blk_size = max(xy_max - xy_min)

                # Calculate the perspective transform matrix and apply it
                M = cv2.getPerspectiveTransform(src, (corners - xy_min).astype(np.float32).reshape(-1, 2))
                warped = cv2.warpPerspective(text_img, M, (blk_size, blk_size), flags=cv2.INTER_CUBIC)
                warped = np.clip(warped, 0, 255)

                warped = warped.swapaxes(0, 1)

                img_blk = image[xy_min[0]:xy_min[0]+warped.shape[0],
                                xy_min[1]:xy_min[1]+warped.shape[1]]

                text_mask = warped > 0

                img_blk[text_mask] = warped[text_mask]

            else:
                c = (36, 255, 12)
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, h, thickness)
                text_w, text_h = text_size

                text_img = np.zeros((text_h + 1, text_w + 1, 3), dtype=image.dtype)
                cv2.putText(text_img, label, (0, text_h), cv2.FONT_HERSHEY_PLAIN, h, c, thickness, cv2.LINE_AA)
                text_img = text_img.transpose((1, 0, 2))

                img_blk = image[max(0, pt[0]-text_img.shape[0]//2):pt[0]+text_img.shape[0]//2,
                                max(0, pt[1]-text_img.shape[1]//2):pt[1]+text_img.shape[1]//2]
                text_img = text_img[:img_blk.shape[0], :img_blk.shape[1]]
                text_mask = text_img > 0

                img_blk[text_mask] = text_img[text_mask]
                # img_blk[:] = text_img

    def _update_image(self):
        image = self.current_frame

        if self._show_keypoints:
            if self.current_frame_idx in self.keypoints_of_frame:
                image = image.copy()
                if self._show_blocks and 'blocks' in self.keypoints_of_frame[self.current_frame_idx]:
                    self._draw_blocks(image, self.keypoints_of_frame[self.current_frame_idx]['blocks'])
                self._draw_keypoints(image, self.keypoints_of_frame[self.current_frame_idx]['keypoints'], self._selected_keypoints)

        self.view.setImage(image, autoRange=False)

    def show_frame(self, frame_idx: int, force_update: bool = False):
        if self.dataloader is None:
            return

        if not force_update and frame_idx == self.current_frame_idx:
            return

        self._selected_keypoints = []

        self.current_frame_idx = frame_idx
        self.current_frame = self.dataloader[frame_idx]

        # draw keypoints
        self._update_image()

        self.sigKeyPointChanged.emit(self.current_frame_idx, self.keypoints_of_frame)

    def select_keypoints(self, selected_keypoints: List[Tuple[float, float]], notify: bool = True):
        self._selected_keypoints = [tuple(pt) for pt in selected_keypoints]

        if notify:
            self._update_image()

    def show_keypoints(self, enabled: bool):
        self._show_keypoints = enabled
        self._update_image()

    def show_heatmap(self, enabled: bool):
        self._show_heatmap = enabled

    def show_blocks(self, enabled: bool, warped: bool):
        self._show_blocks = enabled
        self.blk_text_wrap = warped
        self._update_image()

    @pyqtSlot(float, float, Qt.KeyboardModifiers)
    def on_image_click_event(self, x: float, y: float, modifiers: Qt.KeyboardModifiers):
        if modifiers & Qt.ControlModifier:
            if modifiers & Qt.ShiftModifier:
                self.remove_closed_keypoint(x, y)
            else:
                self.add_keypoint(x, y)
        else:
            self.select_closed_keypoint(x, y)

    # @pyqtSlot(float, float, Qt.KeyboardModifiers)
    # def on_image_dragged_event(self, x: float, y: float, modifiers: Qt.KeyboardModifiers):
    #     if modifiers == Qt.KeyboardModifier.ControlModifier:
    #         for _pt in self._selected_keypoints:
    #             z = list(_pt)
    #             self.keypoints_of_frame[self.current_frame_idx]['keypoints'].remove(_pt)
    #             self._selected_keypoints.remove(_pt)
    #             z[0] = x
    #             z[1] = y
    #             _pt = tuple(z)
    #             self.keypoints_of_frame[self.current_frame_idx]['keypoints'].append(tuple(z))
    #         self._update_image()
    #         self.sigKeyPointChanged.emit(self.current_frame_idx, self.keypoints_of_frame)

    @pyqtSlot(float, float, Qt.KeyboardModifiers)
    def on_image_double_click_event(self, x: float, y: float, modifiers: Qt.KeyboardModifiers):
        # self.add_block()
        return

    def add_keypoint(self, x: float, y: float, notify=True):
        if not self.current_frame_idx in self.keypoints_of_frame:
            self.keypoints_of_frame[self.current_frame_idx] = {'keypoints': [(x, y)], 'blocks': []}

        elif (x, y) in self.keypoints_of_frame[self.current_frame_idx]['keypoints']:
            return

        else:
            self.keypoints_of_frame[self.current_frame_idx]['keypoints'].append((x, y))

        self._selected_keypoints = [(x, y)]

        if notify:
            self._update_image()
            self.sigKeyPointChanged.emit(self.current_frame_idx, self.keypoints_of_frame)

    def _propose_select_threshold(self):
        if self.dataloader is not None:
            threshold = max(1, self.dataloader.width // 200)
        else:
            threshold = 3

        return threshold

    def get_closed_keypoint(self, x: float, y: float, threshold: float = None, num_points=1):
        if threshold is None:
            threshold = self._propose_select_threshold()

        if not self.current_frame_idx in self.keypoints_of_frame:
            return []

        distances = [(pt[0] - x)**2 + (pt[1] - y)**2 for pt in self.keypoints_of_frame[self.current_frame_idx]['keypoints']]

        threshold2 = threshold*threshold
        indices = np.argsort(distances)[:num_points]
        points = [(idx, distances[idx], self.keypoints_of_frame[self.current_frame_idx]['keypoints'][idx]) for idx in indices if distances[idx] < threshold2]

        return points

    @pyqtSlot(float, float)
    def select_closed_keypoint(self, x: float, y: float, threshold: float = None, num_points=1, replace=False):
        ret = self.get_closed_keypoint(x, y, threshold=threshold, num_points=num_points)
        if ret is None or len(ret) == 0:
            self._selected_keypoints.clear()
        else:
            if replace:
                self._selected_keypoints.clear()

            for _idx, _distance, _pt in ret:
                if _pt in self._selected_keypoints:
                    self._selected_keypoints.remove(_pt)
                else:
                    self._selected_keypoints.append(tuple(_pt))
            # print(self._selected_keypoints)
        self._update_image()

    @pyqtSlot(float, float)
    def remove_closed_keypoint(self, x: float, y: float, threshold: float = None):
        ret = self.get_closed_keypoint(x, y, threshold=threshold)
        if ret is None or len(ret) == 0:
            return

        _idx, _distance, _pt = ret[0]
        self.remove_keypoint(_pt[0], _pt[1])

    @pyqtSlot()
    def remove_selected_keypoint(self):
        selected_keypoints = self._selected_keypoints[:]
        for pt in selected_keypoints:
            self.remove_keypoint(pt[0], pt[1], notify=False)

        self._update_image()
        self.sigKeyPointChanged.emit(self.current_frame_idx, self.keypoints_of_frame)

    @pyqtSlot(float, float)
    def remove_keypoint(self, x: float, y: float, notify=True):
        if not self.current_frame_idx in self.keypoints_of_frame:
            raise IndexError

        pt = (x, y)
        self.keypoints_of_frame[self.current_frame_idx]['keypoints'].remove(pt)

        # remove pt from selection
        if pt in self._selected_keypoints:
            self._selected_keypoints.remove(pt)

        # remove all blocks containing pt
        if 'blocks' in self.keypoints_of_frame[self.current_frame_idx]:
            blocks = self.keypoints_of_frame[self.current_frame_idx]['blocks']

            new_block = [blk for blk in blocks if not pt in blk['corners']]
            if len(new_block) != blocks:
                self.keypoints_of_frame[self.current_frame_idx]['blocks'] = new_block

        if len(self.keypoints_of_frame[self.current_frame_idx]['keypoints']) == 0:
            self.keypoints_of_frame.pop(self.current_frame_idx)

        if notify:
            self._update_image()
            self.sigKeyPointChanged.emit(self.current_frame_idx, self.keypoints_of_frame)

    def _get_selected_block_keypoints(self):
        if len(self._selected_keypoints) != 4:
            QMessageBox.warning(self, 'Failed to Add Block Label', f'Four keypoints need to be selected, current: {len(self._selected_keypoints)}')
            return None

        keypoints = tuple(tuple(pt) for pt in sorted(self._selected_keypoints))
        point_order = utils.check_convex_4_points(keypoints)
        if point_order[0] == -1:
            QMessageBox.warning(self, 'Failed to Add Block Label', 'The four selected keypoints cannot form a convex quadrilateral')
            return

        assert point_order[0] == 0
        keypoints = tuple(keypoints[i] for i in point_order)
        return keypoints

    @pyqtSlot()
    def add_block(self, block_label: str | None = None, block_direction: int | None = None, notify=True):
        keypoints = self._get_selected_block_keypoints()
        if keypoints is None:
            return

        if not 'blocks' in self.keypoints_of_frame[self.current_frame_idx]:
            self.keypoints_of_frame[self.current_frame_idx]['blocks'] = []

        blocks = self.keypoints_of_frame[self.current_frame_idx]['blocks']
        used_labels = [blk['label'] for blk in blocks]

        keypoints_np = np.array(keypoints)

        margin = 5
        h, w = self.current_frame.shape[:2]
        xy_min = np.maximum(0, np.floor(keypoints_np.min(axis=0)).astype(int) - margin)
        xy_max = np.minimum((h - 1, w - 1), np.ceil(keypoints_np.max(axis=0)).astype(int) + margin)
        image = self.current_frame[xy_min[0]:xy_max[0]+1, xy_min[1]:xy_max[1]+1]

        # Set up the destination points for the perspective transform
        blk_size = max(image.shape[:2])
        dst = np.array([
            [0, 0],
            [blk_size - 1, 0],
            [blk_size - 1, blk_size - 1],
            [0, blk_size - 1]], dtype=np.float32)

        # Calculate the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform((keypoints_np - xy_min).astype(np.float32).reshape(-1, 2), dst)
        warped = cv2.warpPerspective(image.swapaxes(0, 1), M, (blk_size, blk_size), flags=cv2.INTER_CUBIC).swapaxes(0, 1)
        warped = np.clip(warped, 0, image.max())

        # check if exist
        current_block_idx = -1
        label = ('*' if self.last_blk_label is None else self.last_blk_label) if block_label is None else block_label
        direction = (0 if self.last_blk_direction is None else self.last_blk_direction) if block_direction is None else block_direction

        keypoints_sorted = sorted(keypoints)
        for idx in range(len(blocks)):
            if keypoints_sorted == sorted(blocks[idx]['corners']):
                if blocks[idx]['label'] != '':
                    label = blocks[idx]['label']
                    direction = blocks[idx]['direction']
                current_block_idx = idx
                break

        if block_label is None or block_direction is None:  # if either information is not provided, we need to create
            annotator = BlockAnnotatorWidget(self, warped, label, direction,
                                             label_candidates=self.label_candidates,
                                             image_scale_log=self.last_image_scale_log,
                                             used_labels=used_labels)
            ret = annotator.exec()
            if ret == QDialog.DialogCode.Rejected or annotator.direction < 0:
                return

            label = annotator.label
            direction = annotator.direction
            self.last_image_scale_log = annotator.image_scale_log

        if label != '':
            self.last_blk_label = label
            self.last_blk_direction = direction

        ########
        # capitalize label
        label = str.upper(label)

        if notify:
            quote = '[]'
            print(f'added block: label: {label if len(label) > 0 else quote} direction: {direction}')

        if current_block_idx >= 0:
            blocks[current_block_idx]['corners'] = keypoints
            blocks[current_block_idx]['label'] = label
            blocks[current_block_idx]['direction'] = direction

        else:
            blocks.append({
                'corners': keypoints,
                'label': label,
                'direction': direction,
            })

        if notify:
            self._update_image()

    @pyqtSlot()
    def remove_block(self):
        if len(self._selected_keypoints) != 4:
            QMessageBox.warning(self, 'Failed to Add Block Label', f'Four keypoints need to be selected, current: {len(self._selected_keypoints)}')
            return

        if not self.current_frame_idx in self.keypoints_of_frame:
            return

        if not 'blocks' in self.keypoints_of_frame[self.current_frame_idx]:
            return

        keypoints = tuple(tuple(pt) for pt in sorted(self._selected_keypoints))
        keypoints_sorted = sorted(keypoints)

        blocks: list = self.keypoints_of_frame[self.current_frame_idx]['blocks']
        to_remove = -1
        for idx in range(len(blocks)):
            if keypoints_sorted == sorted(blocks[idx]['corners']):
                to_remove = idx
        if to_remove >= 0:
            blocks.pop(to_remove)

        self._update_image()

    def propagate_block_labels(self):
        if self.label_candidates is None:
            print('no label candidates')
            QMessageBox.warning(self, 'Failed to Propagate Block Label', f'Label candidates need to be loaded!')

            return

        if not 'blocks' in self.keypoints_of_frame[self.current_frame_idx]:
            print('no blocks created')
            QMessageBox.warning(self, 'Failed to Propagate Block Label', f'The selected block has not been created!')
            return

        blocks = self.keypoints_of_frame[self.current_frame_idx]['blocks']

        keypoints = self._get_selected_block_keypoints()
        if keypoints is None:
            return

        keypoints_np = np.array(keypoints)

        current_block_idx = -1
        keypoints_sorted = sorted(keypoints)
        for idx in range(len(blocks)):
            if keypoints_sorted == sorted(blocks[idx]['corners']):
                label = blocks[idx]['label']
                direction = blocks[idx]['direction']
                current_block_idx = idx
                break

        if current_block_idx < 0:
            print('the selected block has not been created')
            QMessageBox.warning(self, 'Failed to Propagate Block Label', f'The selected block has not been created!')
            return

        if label in ['']:
            print('the selected block has not been labeled')
            QMessageBox.warning(self, 'Failed to Propagate Block Label', f'The selected block has not been labeled!')
            return

        if label in ['*']:
            print('the selected block should be a labeled non-black block')
            QMessageBox.warning(self, 'Failed to Propagate Block Label', f'The selected block should be a labeled non-black block!')
            return

        label_idx = utils.find_index_in_nested_list(self.label_candidates, label)
        if label_idx is None:
            print(f'label {label} cannot be found in the candidate list, something must be wrong...')
            QMessageBox.warning(self, 'Failed to Propagate Block Label',
                                f'Label {label} cannot be found in the candidate list, something must be wrong...')
            return

        keypoints_np = np.roll(keypoints_np, shift=-direction, axis=0)

        ################
        #  marker index:
        #  0._______.1
        #   |       |
        #   |       |
        #   |_______|
        #  3         2
        # note this is *different* from that in `test_triangulation.py`
        ################

        visited_blocks = [current_block_idx]

        def propagate_label(curr_blk_corners: np.ndarray, curr_label_idx: Tuple[int, int]):
            edges = [
                # shared edge points,  direction,  idx offset
                (curr_blk_corners[[0, 3]], 1, (0, -1)),  # the left wall
                (curr_blk_corners[[3, 2]], 0, (1, 0)),  # the bottom wall
                (curr_blk_corners[[2, 1]], 3, (0, 1)),  # the right wall
                (curr_blk_corners[[1, 0]], 2, (-1, 0))  # the top wall
            ]

            for edge_points, direction, idx_offset in edges:
                row = curr_label_idx[0] + idx_offset[0]
                col = curr_label_idx[1] + idx_offset[1]
                if row < 0 or row >= len(self.label_candidates):
                    continue
                if len(self.label_candidates[row]) == 0:
                    continue
                if col < 0 or col >= len(self.label_candidates[row]):
                    continue
                if self.label_candidates[row][col] == ':':
                    continue

                next_label = self.label_candidates[row][col]
                if next_label == '':
                    next_label = '*'

                next_blk_idx = -1
                next_blk_corners = None
                for idx in range(len(blocks)):
                    if idx in visited_blocks:
                        continue

                    corners = np.array(blocks[idx]['corners'])
                    if edge_points[0] in corners and edge_points[1] in corners:
                        next_blk_curr_label = blocks[idx]['label']
                        next_blk_curr_direction = blocks[idx]['direction']
                        next_blk_idx = idx
                        next_blk_corners = corners
                        break

                if next_blk_idx < 0:
                    continue

                idx = (next_blk_corners == edge_points[:1])[:, 0].nonzero()[0][0]
                next_blk_corners = np.roll(next_blk_corners, direction-idx, axis=0)

                # print(direction, (next_blk_corners == edge_points[:1])[:,0].nonzero()[0][0], f'label coord: {(row, col)}')
                print(f'block {next_blk_idx}: {next_blk_curr_label}, new label: {next_label}')

                blocks[next_blk_idx]['label'] = next_label
                blocks[next_blk_idx]['direction'] = int((-direction+idx) % 4)

                visited_blocks.append(next_blk_idx)
                propagate_label(next_blk_corners, (row, col))

        propagate_label(keypoints_np, label_idx)

        self._update_image()

    def auto_label(self):
        if self.auto_labeler is None:
            return

        # raw_markers, markers, blocks = self.auto_labeler.label(self.current_frame, 0.0, False, 0.0)
        raw_markers, markers, blocks = self.auto_labeler.label(self.current_frame, self.marker_confidence_thr, True, self.edge_confidence_thr, self.marker_distance_thr)
        markers = np.asarray(markers, dtype=float)
        raw_markers = np.asarray(raw_markers, dtype=float)

        # reset current frame...
        self.keypoints_of_frame[self.current_frame_idx] = {'keypoints': [], 'blocks': []}
        self._selected_keypoints = []

        h, w = self.current_frame.shape[:2]
        # add new points
        for mk in markers:
            self.add_keypoint(mk[1], mk[0], notify=False)

        for (area, corner_indices) in blocks:
            self.select_keypoints(markers[list(corner_indices)][:, [1, 0]].tolist(), notify=False)
            self.add_block('', 0, notify=False)

        # add raw markers that are not used by blocks
        distance = scipy.spatial.distance_matrix(raw_markers, np.concatenate((raw_markers, markers), axis=0))
        np.fill_diagonal(distance, 1e20)
        min_distance = distance.min(axis=-1)
        distance_thr = 10  # px
        unique_marker_indices = np.nonzero(min_distance > distance_thr)[0]
        for mk in raw_markers[unique_marker_indices]:
            self.add_keypoint(mk[1], mk[0], notify=False)

        self._update_image()
        self.sigKeyPointChanged.emit(self.current_frame_idx, self.keypoints_of_frame)

        return len(raw_markers), len(markers), len(blocks)


class Window(QMainWindow):
    def __init__(self):
        super().__init__(parent=None)
        self.setWindowTitle("Annotator")

        self.annotator = AnnotatorWidget()
        self.setCentralWidget(self.annotator)

        self.keypoint_file = None

        self.recent_configs = []

        # timeline
        self._createTimeLine()

        # keypoints
        self._createKeyPointViews()

        # menu
        self._createMenu()

        self.status = QStatusBar()
        self.status.showMessage('Init')
        self.setStatusBar(self.status)

    def _createTimeLine(self):
        timeline_toolbar = QToolBar()
        timeline_toolbar.setWindowTitle('Timeline')
        timeline_toolbar.setAllowedAreas(Qt.ToolBarArea.TopToolBarArea | Qt.ToolBarArea.BottomToolBarArea)
        timeline_toolbar.setStyleSheet('border: 1px solid black')
        self.timeline = TimelineWidget()
        timeline_toolbar.addWidget(self.timeline)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, timeline_toolbar)

        self.timeline.frameSlider.valueChanged.connect(self.annotator.show_frame)

    def _createKeyPointViews(self):
        # keypoints
        self.keypoint_view = pg.TableWidget(editable=False, sortable=True)
        self.keypoint_view.setFormat('%0.3f')

        keypoint_widget = QDockWidget('KeyPoints')
        keypoint_widget.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        keypoint_widget.setMinimumWidth(300)
        keypoint_widget.setWidget(self.keypoint_view)

        def update_keypoints(curr_frame: int, keypoints: Dict[int, Dict[str, Union[List[Tuple[int, int]], Any]]]):
            if curr_frame in keypoints:
                keypoint_list = keypoints[curr_frame]['keypoints']
            else:
                keypoint_list = []
            self.keypoint_view.setData(keypoint_list)

        self.annotator.sigKeyPointChanged.connect(update_keypoints)

        def remove_keypoint():
            selection = self.keypoint_view.selectedRanges()[0]
            rows = list(range(selection.topRow(),
                              selection.bottomRow() + 1))
            to_remove = [(self.keypoint_view.item(r, 0).value, self.keypoint_view.item(r, 1).value) for r in rows]
            for (x, y) in to_remove:
                self.annotator.remove_keypoint(x, y)

        menu: QMenu = self.keypoint_view.contextMenu
        menu.addSeparator()
        menu.addAction('Remove').triggered.connect(remove_keypoint)

        def update_anno_selection():
            if len(self.keypoint_view.selectedRanges()) == 0:
                self.annotator.select_keypoints([])
                return

            selection = self.keypoint_view.selectedRanges()[0]
            rows = list(range(selection.topRow(),
                              selection.bottomRow() + 1))
            selected = [(self.keypoint_view.item(r, 0).value, self.keypoint_view.item(r, 1).value) for r in rows]
            self.annotator.select_keypoints(selected)

        self.keypoint_view.itemSelectionChanged.connect(update_anno_selection)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, keypoint_widget)

        # keypointed frames
        self.keypointed_frames_view = pg.TableWidget(editable=False, sortable=True)

        keypoint_widget = QDockWidget('KeyPointed Frames')
        keypoint_widget.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        keypoint_widget.setMinimumWidth(200)
        keypoint_widget.setWidget(self.keypointed_frames_view)

        def update_keypointed_frames(curr_frame: int, keypoints: Dict[int, Dict[str, Union[List[Tuple[int, int]], Any]]]):
            curr_list = [self.keypointed_frames_view.item(r, 0).value for r in range(self.keypointed_frames_view.rowCount())]
            curr_list = sorted(curr_list)

            new_list = sorted(list(keypoints.keys()))
            if curr_list != new_list:
                self.keypointed_frames_view.setData(new_list)

        self.annotator.sigKeyPointChanged.connect(update_keypointed_frames)

        def frame_double_clicked(row: int, col: int):
            item = self.keypointed_frames_view.item(row, col)
            if item is not None:
                self.timeline.setValue(item.value)
                # self.annotator.show_frame(item.value)

        self.keypointed_frames_view.cellDoubleClicked.connect(frame_double_clicked)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, keypoint_widget)

    def _createMenu(self):
        # File menu
        menu = self.menuBar().addMenu("&File")
        menu.addAction("&Open...", self.open, shortcut='Ctrl+O')
        menu.addSeparator()
        menu.addAction("&Load...", self.load, shortcut='Ctrl+L')
        menu.addAction("&Save...", self.save, shortcut='Ctrl+S')
        menu.addAction("Save As...", self.saveAs)
        menu.addSeparator()
        menu.addAction("&Export...", self.export)
        menu.addSeparator()
        menu.addAction("&Save Config...", self.save_config)
        menu.addAction("&Load Config...", self.load_config)
        recentConfigsMenu = menu.addMenu("&Recent Configs...")
        menu.addSeparator()
        menu.addAction("&Exit", self.close)

        def selectRecentConfig():
            for a in recentConfigsMenu.actions():
                if a.hovered:
                    self.selectedRecentConfigAction = a
                    break

        def loadSelectedRecentConfig():
            if self.selectedRecentConfigAction:
                if self.selectedRecentConfigAction.text != '':
                    self.load_config(self.selectedRecentConfigAction.text())

        def fillRecentConfigs():
            path = os.path.join('', '_configs')
            files = ''
            if os.path.isdir(path):
                files = os.scandir(path)
            recentConfigsMenu.clear()
            noneAction = recentConfigsMenu.addAction('NONE')
            noneAction.setEnabled(False)

            if files != '':
                recentConfigsMenu.clear()
                for file in files:
                    a = recentConfigsMenu.addAction(os.path.join(path, file.name))
                    a.hovered.connect(selectRecentConfig)
                    a.triggered.connect(loadSelectedRecentConfig)

        recentConfigsMenu.aboutToShow.connect(fillRecentConfigs)

        def menuAnnotationUpdate():
            self.heatmapAction.setChecked(self.annotator._show_heatmap)
            self.loadKeypointModelAction.setText(f'Load Keypoint Model - Current: {self.annotator.keypoint_model}')
            self.loadEdgeModelAction.setText(f'Load Edge Model - Current: {self.annotator.edge_model}')

            self.setMarkerDistanceThrAction.setText(f'Set Marker Distance Threshold - Current: {self.annotator.marker_distance_thr}')
            self.setMarkerConfidenceThrAction.setText(f'Set Marker Confidence Threshold - Current: {self.annotator.marker_confidence_thr}')
            self.setEdgeConfidenceThrAction.setText(f'Set Edge Confidence Threshold - Current: {self.annotator.edge_confidence_thr}')

        def menuViewUpdate():
            self.viewAction.setChecked(self.annotator._show_keypoints)
            self.viewKPSizeAction.setText(f'Keypoint Size - Radius: {self.annotator.kp_radius} | Thickness: {self.annotator.kp_thickness}')
            self.viewBlkAction.setChecked(self.annotator._show_blocks)
            self.viewBlkTextSizeAction.setText(f'Block Text Size - Text Height: {self.annotator.blk_text_height} | Thickness: {self.annotator.blk_text_thickness}')
            self.viewBlkWarpAction.setChecked(self.annotator.blk_text_wrap)

        # Edit menu
        menu = self.menuBar().addMenu("&Edit")

        # selectAllAction = menu.addAction('Select All')
        # selectAllAction.setShortcut('CTRL+A')
        # selectAllAction.setVisible(True)
        # def select_all():
        #    self.annotator._selected_keypoints = self.annotator.keypoints_of_frame[self.annotator.current_frame_idx]['keypoints']
        #    self.annotator._update_image()

        # selectAllAction.triggered.connect(select_all)

        deselectAllAction = menu.addAction('Deselect All')
        deselectAllAction.setShortcuts(['Space', 'Escape'])
        deselectAllAction.setVisible(True)

        def deselect_all():
            self.annotator._selected_keypoints = []
            self.annotator._update_image()

        deselectAllAction.triggered.connect(deselect_all)

        menu.addSeparator()

        removeKeypointAction = menu.addAction('>    &Remove Keypoints')
        removeKeypointAction.setVisible(True)
        removeKeypointAction.setShortcuts({'Del', 'X'})
        removeKeypointAction.triggered.connect(self.annotator.remove_selected_keypoint)

        addBlockAction = menu.addAction('>    Add &Block')
        addBlockAction.setVisible(True)
        addBlockAction.setShortcuts({'B', 'Ctrl+B'})
        addBlockAction.triggered.connect(self.annotator.add_block)

        markBlockAction = menu.addAction('>    &Mark Block')
        markBlockAction.setVisible(True)
        markBlockAction.setShortcuts({'G', 'Ctrl+G'})
        markBlockAction.triggered.connect(partial(self.annotator.add_block, block_label='', block_direction=0))

        addBlackBlockAction = menu.addAction('>    Add Bla&ck Block')
        addBlackBlockAction.setVisible(True)
        addBlackBlockAction.setShortcuts({'D', 'Ctrl+D'})
        addBlackBlockAction.triggered.connect(partial(self.annotator.add_block, block_label='*', block_direction=0))

        removeBlockAction = menu.addAction('>    &Remove Block')
        removeBlockAction.setVisible(True)
        removeBlockAction.setShortcuts({'Ctrl+Del', 'Ctrl+X'})
        removeBlockAction.triggered.connect(self.annotator.remove_block)

        propagateBlockAction = menu.addAction('>    &Propagate Block Labels')
        propagateBlockAction.setVisible(True)
        propagateBlockAction.triggered.connect(self.propagate_block_labels)

        # Annotation menu
        menu = self.menuBar().addMenu("&Annotation")
        menu.aboutToShow.connect(menuAnnotationUpdate)

        autoLabelKeypointAction = menu.addAction('Auto Label Keypoints')
        autoLabelKeypointAction.setShortcut('A')
        autoLabelKeypointAction.setVisible(True)

        autoLabelKeypointAction.triggered.connect(self.auto_label_keypoints)

        self.heatmapAction = menu.addAction('>    Show Heatmap')
        self.heatmapAction.setShortcut('H')
        self.heatmapAction.setCheckable(True)
        self.heatmapAction.setChecked(True)

        def show_heatmap(enabled):
            self.annotator.show_heatmap(enabled)
            self.status.showMessage(f'Show Heatmap: {enabled}')
            if self.annotator.auto_labeler != None:
                self.annotator.auto_labeler.show_heatmap = enabled
        self.heatmapAction.triggered.connect(show_heatmap)

        menu.addSeparator()

        self.loadKeypointModelAction = menu.addAction(f'Load Keypoint Model - Current: {self.annotator.keypoint_model}')
        self.loadKeypointModelAction.setVisible(True)

        def load_keypoint_model():
            self.annotator.keypoint_model = self.load_model('Load Keypoint Model')
            self.annotator.auto_labeler = None
        self.loadKeypointModelAction.triggered.connect(load_keypoint_model)

        self.loadEdgeModelAction = menu.addAction(f'Load Edge Model - Current: {self.annotator.edge_model}')
        self.loadEdgeModelAction.setVisible(True)

        def load_edge_model():
            self.annotator.edge_model = self.load_model('Load Edge Model')
            self.annotator.auto_labeler = None
        self.loadEdgeModelAction.triggered.connect(load_edge_model)

        menu.addSeparator()

        self.setMarkerDistanceThrAction = menu.addAction(f'Set Marker Distance Threshold - Current: {self.annotator.marker_distance_thr}')
        self.setMarkerDistanceThrAction.setVisible(True)

        def set_marker_distance_threshold():
            val, ret1 = QInputDialog.getInt(self, 'Marker Distance Threshold', 'PX-Distance (def.: 65)', self.annotator.marker_distance_thr)
            if ret1:
                self.annotator.marker_distance_thr = val

        self.setMarkerDistanceThrAction.triggered.connect(set_marker_distance_threshold)

        self.setMarkerConfidenceThrAction = menu.addAction(f'Set Marker Confidence Threshold - Current: {self.annotator.marker_confidence_thr}')
        self.setMarkerConfidenceThrAction.setVisible(True)

        def set_marker_confidence_threshold():
            val, ret1 = QInputDialog.getDouble(self, 'Marker Confidence Threshold', 'Threshold (def.: 0.75)', self.annotator.marker_confidence_thr, decimals=4)
            if ret1:
                self.annotator.marker_confidence_thr = val

        self.setMarkerConfidenceThrAction.triggered.connect(set_marker_confidence_threshold)

        self.setEdgeConfidenceThrAction = menu.addAction(f'Set Edge Confidence Threshold - Current: {self.annotator.edge_confidence_thr}')
        self.setEdgeConfidenceThrAction.setVisible(True)

        def set_edge_confidence_threshold():
            val, ret1 = QInputDialog.getDouble(self, 'Edge Confidence Threshold', 'Threshold (def.: 0.75)', self.annotator.edge_confidence_thr, decimals=4)
            if ret1:
                self.annotator.edge_confidence_thr = val

        self.setEdgeConfidenceThrAction.triggered.connect(set_edge_confidence_threshold)

        loadLabelCandidatesAction = menu.addAction('>    Load Label &Candidate')
        loadLabelCandidatesAction.setVisible(False)
        loadLabelCandidatesAction.triggered.connect(self.load_label_candidates)

        menu = self.menuBar().addMenu("&View")
        menu.aboutToShow.connect(menuViewUpdate)

        # refreshAction = menu.addAction('Refresh')
        # refreshAction.setShortcut('F5')
        # refreshAction.setVisible(True)

        # def refresh():
        #     self.annotator._update_image()
        # refreshAction.triggered.connect(refresh)

        menu.addSeparator()
        self.viewAction = menu.addAction('Show &Keypoints')
        self.viewAction.setShortcut('Ctrl+M')
        self.viewAction.setCheckable(True)
        self.viewAction.setChecked(True)

        def show_keypoints(enabled):
            self.annotator.show_keypoints(enabled)
        self.viewAction.triggered.connect(show_keypoints)

        self.viewKPSizeAction = menu.addAction(f'Keypoint Size - Radius: {self.annotator.kp_radius} | Thickness: {self.annotator.kp_thickness}')

        def set_kp_sizes():
            val, ret1 = QInputDialog.getInt(self, 'Keypoint Size', 'Radius', self.annotator.kp_radius)
            if ret1:
                self.annotator.kp_radius = val
            val, ret2 = QInputDialog.getInt(self, 'Keypoint Size', 'Thickness', self.annotator.kp_thickness)
            if ret2:
                self.annotator.kp_thickness = val

            if ret1 or ret2:
                self.annotator._update_image()

        self.viewKPSizeAction.triggered.connect(set_kp_sizes)

        self.viewBlkTextSizeAction = menu.addAction(f'Block Text Size - Text Height: {self.annotator.blk_text_height} | Thickness: {self.annotator.blk_text_thickness}')

        def set_kp_sizes():
            val, ret1 = QInputDialog.getDouble(self, 'Block Text Size', 'Text Height', self.annotator.blk_text_height)
            if ret1:
                self.annotator.blk_text_height = val
            val, ret2 = QInputDialog.getInt(self, 'Block Text Size', 'Thickness', self.annotator.blk_text_thickness)
            if ret2:
                self.annotator.blk_text_thickness = val

            if ret1 or ret2:
                self.annotator._update_image()

        self.viewBlkTextSizeAction.triggered.connect(set_kp_sizes)

        self.viewBlkAction = menu.addAction('Show B&locks')
        self.viewBlkAction.setShortcut('Tab')
        self.viewBlkAction.setCheckable(True)
        self.viewBlkAction.setChecked(True)

        self.viewBlkWarpAction = menu.addAction('>    Show Blocks &Warped')
        self.viewBlkWarpAction.setShortcut('Ctrl+Shift+W')
        self.viewBlkWarpAction.setVisible(True)
        self.viewBlkWarpAction.setCheckable(True)
        self.viewBlkWarpAction.setChecked(self.annotator.blk_text_wrap)

        def show_blocks(enabled):
            self.annotator.show_blocks(enabled, self.annotator.blk_text_wrap)
            self.viewBlkWarpAction.setVisible(enabled)
        self.viewBlkAction.triggered.connect(show_blocks)

        def show_warped_blocks(enabled):
            self.annotator.show_blocks(True, warped=enabled)
        self.viewBlkWarpAction.triggered.connect(show_warped_blocks)

        menu = self.menuBar().addMenu("&Help")

        def show_help():
            QMessageBox.about(self, "About", AnnotatorWidget.help_msg)
        menu.addAction('&About', show_help)

    def load(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, 'Load Annotation',
                                                     filter=('Annotation files (*.json);;All files (*.*)'))

        if len(file_names) == 0:
            return

        self.annotator.load_keypoints(file_names)

        if len(file_names) == 1:
            self.keypoint_file = file_names[0]
        else:
            self.keypoint_file = None

    def load_label_candidates(self):
        fn, _ = QFileDialog.getOpenFileName(self, 'Load Label Candidates',
                                            filter=('Label files (*.json);;All files (*.*)'))

        if fn == '':
            return

        self.annotator.load_label_candidates(fn)

    def auto_label_keypoints(self):
        btn = QMessageBox.warning(self, 'Auto Label Warning', 'This operation will remove current labels, continue?',
                                  QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Ok)

        if btn != QMessageBox.StandardButton.Ok:
            return

        if self.annotator.auto_labeler is None:
            if self.annotator.keypoint_model == '':
                self.annotator.keypoint_model = self.load_model('Load Keypoint Model')

            if self.annotator.edge_model == '':
                self.annotator.edge_model = self.load_model('Load Edge Model')

            if self.annotator.keypoint_model == '' or self.annotator.edge_model == '':
                btn = QMessageBox.critical(self, 'Failed', 'Failed to setup auto labeler!', QMessageBox.StandardButton.Close, QMessageBox.StandardButton.Close)
                return
            else:
                self.annotator.auto_labeler = CornerLabeler(self.annotator.keypoint_model, self.annotator.edge_model, device='cuda', show_heatmap=self.annotator._show_heatmap)

        raw_markers, markers, blocks = self.annotator.auto_label()
        QMessageBox.information(self, 'Auto Label Keypoints', f'Raw Markers: {raw_markers}, Refined Markers: {markers}, Blocks: {blocks}', QMessageBox.StandardButton.Ok, QMessageBox.StandardButton.Ok)

    def load_model(self, title: str = 'Load Model', filter: str = ('Model (*.pth *.pt);;All files (*.*)')):
        model_fn, _ = QFileDialog.getOpenFileName(self, title, filter=filter)
        self.status.showMessage(f'Loaded Model: {model_fn}')
        return model_fn

    def propagate_block_labels(self):
        btn = QMessageBox.warning(self, 'Block Label Propagation Warning', 'This operation will override some labels, continue?',
                                  QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)

        if btn != QMessageBox.StandardButton.Ok:
            return

        self.annotator.propagate_block_labels()

    def saveAs(self):
        self.save(save_as=True)

    def save(self, save_as: bool = False):
        if self.keypoint_file is None or save_as:
            fn, _ = QFileDialog.getSaveFileName(self, 'Load Annotation',
                                                filter=('Annotation files (*.json);;All files (*.*)'))

            if fn == '':
                return

            self.keypoint_file = fn

        self.annotator.save_keypoints(self.keypoint_file)

    def export(self):
        folder = QFileDialog.getExistingDirectory(self, 'Export Annotation')

        if folder == '':
            return

        self.annotator.export(folder)

    def load_config(self, fn: str = ''):
        if fn == '':
            fn, typ = QFileDialog.getOpenFileName(self, 'Open Config',
                                                  filter=('Config files (*.json);;All files (*.*)'))

        if len(fn) <= 0:
            return

        with open(fn, 'r') as f:
            data = json.load(f)

            self.annotator.keypoint_model = data[0]["keypoint_model"]
            self.annotator.edge_model = data[0]["edge_model"]
            self.annotator.auto_labeler = None

            self.annotator.marker_distance_thr = data[0]["marker_distance_thr"]
            self.annotator.marker_confidence_thr = data[0]["marker_confidence_thr"]
            self.annotator.edge_confidence_thr = data[0]["edge_confidence_thr"]

            self.annotator.show_heatmap(data[0]["show_heatmap"])
            self.annotator.show_keypoints(data[0]["show_keypoints"])
            self.annotator.show_blocks(data[0]["show_blocks"], data[0]["blk_text_wrap"])
            self.annotator.kp_radius = data[0]["kp_radius"]
            self.annotator.kp_thickness = data[0]["kp_thickness"]
            self.annotator.blk_text_height = data[0]["blk_text_height"]
            self.annotator.blk_text_thickness = data[0]["blk_text_thickness"]

            if "files" in data[0]:
                self.open(data[0]["files"])
            if "recent_configs" in data[0]:
                self.recent_configs = data[0]["recent_configs"]

            self.status.showMessage(f'Loaded config: "{fn}"')
            print(f'\tLoaded config: "{fn}"')
            return

        QMessageBox.warning(self, 'Error', f'An error occured!')

    def save_config(self, fn: str = ''):
        if fn == '':
            fn, typ = QFileDialog.getSaveFileName(self, 'Save Config',
                                                  filter=('Config files (*.json);;All files (*.*)'))

        if len(fn) <= 0:
            return

        files = []

        if self.annotator.dataloader:
            files = self.annotator.dataloader.fn

        data = [{'keypoint_model': self.annotator.keypoint_model,
                 'edge_model': self.annotator.edge_model,
                 'marker_distance_thr': self.annotator.marker_distance_thr,
                 'marker_confidence_thr': self.annotator.marker_confidence_thr,
                 'edge_confidence_thr': self.annotator.edge_confidence_thr,
                 'show_heatmap': self.annotator._show_heatmap,
                 'show_keypoints': self.annotator._show_keypoints,
                 'kp_radius': self.annotator.kp_radius,
                 'kp_thickness': self.annotator.kp_thickness,
                 'show_blocks': self.annotator._show_blocks,
                 'blk_text_height': self.annotator.blk_text_height,
                 'blk_text_thickness': self.annotator.blk_text_thickness,
                 'blk_text_wrap': self.annotator.blk_text_wrap,
                 'files': files,
                 'recent_configs': self.recent_configs
                 }]

        with open(fn, 'w') as f:
            json.dump(data, f, indent=4)

        self.status.showMessage(f'Saved config: "{fn}"')
        print(f'\tSaved config: "{fn}"')
        return fn

    def open(self, fn: List[str] = []):
        video_suffix = ['mp4', 'avi', 'mov', 'raw']
        image_suffix = ['jpg', 'png', 'bmp']

        file_names = fn
        typ = ''

        if file_names == []:
            file_names, typ = QFileDialog.getOpenFileNames(self, 'Load Video/Images',
                                                           filter=(f'Image Files ({" ".join("*."+suffix for suffix in image_suffix)});;' +
                                                                   f'Video files ({" ".join("*."+suffix for suffix in video_suffix)});;' +
                                                                   'All files (*.*)'))

        if len(file_names) == 0:
            return

        print(file_names, ':', typ)

        suffix = pathlib.PurePath(file_names[0]).suffix
        suffix = suffix[1:]
        if suffix in video_suffix or len(file_names) > 1:
            self.annotator.open_video(file_names)
        elif suffix in image_suffix:
            self.annotator.open_image(file_names[0])
        else:
            QMessageBox.warning(self, 'Unsupported Format', f'Warning! Unsupported format (.{suffix})')
            return

        self.keypoint_file = None

        if self.annotator.num_images() > 0:
            self.timeline.setRange(0, self.annotator.num_images())
            self.timeline.setValue(0)
            self.annotator.show_frame(0)

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

    window.close()
    sys.exit(code)
