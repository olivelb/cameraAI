"""Native Windows camera application. Run: python desktop.py."""

from __future__ import annotations

import argparse
import html
import logging
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRect, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QImage, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QPlainTextEdit,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QTextBrowser,
    QVBoxLayout, QWidget,
)

import camera_core as core
from config import cfg
from desktop_backend import CommandWorker, camera_settings, restart_camera, save_connection, tailscale_devices
from platform_utils import apply_windows_boost

ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)
NETWORK_PRESETS = {
    "Réglages actuels": ("current", None, None),
    "Réseau local": ("local", 10000, 3000),
    "VPN / réseau mobile": ("vpn", 12000, 10000),
    "Tailscale": ("tailscale", 12000, 10000),
}

STYLE = """
QMainWindow, QWidget { background:#101724; color:#e4eaf3; font:10pt 'Segoe UI'; }
QLabel#brand {font-size:22pt; font-weight:700; color:#f4f8ff;}
QLabel#subtitle {color:#94a6bf;}
QLabel#connection {color:#64dcc0; font-weight:600;}
QTabWidget::pane {border:1px solid #2c3b51; border-radius:8px;}
QTabBar::tab {background:#192438; padding:10px 12px; color:#a8b8cc;}
QTabBar::tab:selected {background:#263b54; color:#7ce6ca;}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextBrowser {
 background:#172237; border:1px solid #34445c; border-radius:5px; padding:7px;
 selection-background-color:#327b89; }
QPushButton {background:#25374e; border:1px solid #3a5069; border-radius:6px; padding:9px 12px;}
QPushButton:hover {background:#324b67;} QPushButton:disabled {color:#68768a; background:#1b2739;}
QPushButton#primary {background:#126e65; border-color:#208b7f; font-weight:600;}
QPushButton#primary:hover {background:#18877c;}
QCheckBox {spacing:8px; padding:4px;} QCheckBox::indicator {width:17px; height:17px;}
QStatusBar {background:#172237; color:#b8c8dd;}
QScrollArea {border:0;} QSplitter::handle {background:#26354b; width:5px;}
"""


def frame_image(frame: np.ndarray) -> QImage:
    """Own the image memory: the capture/AI thread can replace its array safely."""
    pixels = np.ascontiguousarray(frame)
    height, width = pixels.shape[:2]
    return QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_BGR888).copy()


class VideoView(QWidget):
    """Aspect-preserving direct image painting; no JPEG or per-frame event queue."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.image = QImage()
        self.notice = "Connexion à la caméra…"

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#090e17"))
        if not self.image.isNull():
            size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRect((self.width() - size.width()) // 2,
                           (self.height() - size.height()) // 2, size.width(), size.height())
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(target, self.image)
        if self.notice:
            painter.fillRect(0, self.height() - 44, self.width(), 44, QColor(9, 14, 23, 220))
            painter.setPen(QColor("#e4eaf3"))
            painter.drawText(self.rect().adjusted(16, 8, -16, -12),
                             Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, self.notice)
        painter.end()


class CameraWindow(QMainWindow):
    def __init__(self, *, start_camera=True):
        super().__init__()
        self.setWindowTitle("Smart Camera — Vision IA")
        self.setWindowIcon(QIcon(str(ROOT / "assets" / "smart-camera.ico")))
        self.resize(1380, 880)
        self.setMinimumSize(980, 650)
        self.preferences = QSettings("CameraAI", "SmartCamera")
        self.worker = CommandWorker()
        self._completion = None
        self._closing = False
        self._last_image_key = None
        self._painted = 0
        self._metric_time = time.monotonic()
        self._last_log = ""
        self._snapshot = None
        self._history = []
        self._face_files = []
        self._animal_files = []
        self._actions = []
        self._build()
        self.setStyleSheet(STYLE)
        geometry = self.preferences.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.slow_timer = QTimer(self)
        self.slow_timer.setInterval(250)
        self.slow_timer.timeout.connect(self.poll_commands)
        self.slow_timer.start()
        if start_camera:
            core.start_background_threads()
            QTimer.singleShot(300, self.read_settings)

    def button(self, label, callback, *, primary=False, background=True):
        button = QPushButton(label)
        button.clicked.connect(callback)
        if primary:
            button.setObjectName("primary")
        if background:
            self._actions.append(button)
        return button

    def page(self, title):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 18, 16, 18)
        layout.setSpacing(12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.tabs.addTab(scroll, title)
        return layout

    @staticmethod
    def note(text):
        label = QLabel(text)
        label.setObjectName("subtitle")
        label.setWordWrap(True)
        return label

    @staticmethod
    def combo(choices, current=None):
        widget = QComboBox()
        widget.addItems(choices)
        if current:
            widget.setCurrentText(current)
        return widget

    @staticmethod
    def number(low, high, current, *, decimals=None, step=1):
        widget = QSpinBox() if decimals is None else QDoubleSpinBox()
        if decimals is not None:
            widget.setDecimals(decimals)
        widget.setRange(low, high)
        widget.setSingleStep(step)
        widget.setValue(current)
        return widget

    def _build(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(20, 16, 20, 12)
        header = QHBoxLayout()
        title = QLabel("Smart Camera")
        title.setObjectName("brand")
        header.addWidget(title)
        subtitle = self.note("VISION IA  /  POSTE DE CONTRÔLE")
        subtitle.setWordWrap(False)
        header.addWidget(subtitle)
        header.addStretch()
        self.connection = QLabel("Connexion…")
        self.connection.setObjectName("connection")
        header.addWidget(self.connection)
        outer.addLayout(header)
        self.splitter = QSplitter()
        left = QWidget()
        video_layout = QVBoxLayout(left)
        video_layout.setContentsMargins(0, 0, 12, 0)
        self.video = VideoView()
        video_layout.addWidget(self.video, 1)
        tools = QHBoxLayout()
        self.overlay = QCheckBox("Annotations IA")
        self.overlay.setChecked(True)
        self.pause = QCheckBox("Pause de l’affichage")
        tools.addWidget(self.overlay)
        tools.addWidget(self.pause)
        tools.addStretch()
        self.snapshot_button = self.button("Capture…", self.save_snapshot, background=False)
        tools.addWidget(self.snapshot_button)
        tools.addWidget(self.button("Plein écran · F11", self.fullscreen, background=False))
        video_layout.addLayout(tools)
        self.metrics = QLabel("Réception —   ·   Affichage —")
        video_layout.addWidget(self.metrics)
        self.ai_status = self.note("Chargement du moteur IA en arrière-plan…")
        video_layout.addWidget(self.ai_status)
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(410)
        self.tabs.setMaximumWidth(690)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self.tabs)
        self.splitter.setSizes([880, 460])
        outer.addWidget(self.splitter, 1)
        self.setCentralWidget(central)
        self._camera_page()
        self._ai_page()
        self._gallery_page(animal=False)
        self._gallery_page(animal=True)
        self._alerts_page()
        self._assistant_page()
        self._logs_page()
        self.statusBar().showMessage("Prêt — les modèles et la caméra se connectent en arrière-plan")
        full_action = QAction(self)
        full_action.setShortcut(QKeySequence("F11"))
        full_action.triggered.connect(self.fullscreen)
        self.addAction(full_action)
        escape = QAction(self)
        escape.setShortcut(QKeySequence("Escape"))
        escape.triggered.connect(self.showNormal)
        self.addAction(escape)

    def _camera_page(self):
        layout = self.page("Caméra")
        layout.addWidget(self.note("Indique l’adresse de ta caméra ou du point d’accès configuré. Pour un accès distant, active ton VPN avant la connexion."))
        form = QFormLayout()
        self.host = QLineEdit(cfg.pi.host)
        self.video_port = self.number(1, 65535, cfg.pi.video_port)
        self.ws_port = self.number(1, 65535, cfg.pi.ws_port)
        self.token = QLineEdit(cfg.pi.auth_token)
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.open_timeout = self.number(1000, 60000, cfg.stream.capture_open_timeout_ms, step=1000)
        self.read_timeout = self.number(1000, 60000, cfg.stream.capture_read_timeout_ms, step=1000)
        self.network_profile = self.combo(list(NETWORK_PRESETS))
        self.network_profile.currentTextChanged.connect(self.select_network_profile)
        form.addRow("Profil réseau", self.network_profile)
        for title, widget in (("Adresse du Pi / routeur", self.host), ("Port vidéo", self.video_port),
                              ("Port commandes", self.ws_port), ("Jeton de contrôle", self.token),
                              ("Attente connexion (ms)", self.open_timeout), ("Attente lecture (ms)", self.read_timeout)):
            form.addRow(title, widget)
        layout.addLayout(form)
        self.tailscale_button = self.button("Détecter les appareils Tailscale…", self.detect_tailscale)
        self.tailscale_button.setVisible(False)
        layout.addWidget(self.tailscale_button)
        layout.addWidget(self.button("Enregistrer et reconnecter", self.connect_camera, primary=True))
        layout.addWidget(self.button("Lire les réglages du Pi", self.read_settings))
        layout.addWidget(self.button("Relancer le flux du Pi", lambda: self.run("Relance du flux", restart_camera, self.settings_received)))
        settings = QFormLayout()
        self.resolution = self.combo([f"{w}x{h}" for w, h in sorted(cfg.camera.stream_resolution_set())],
                                     "x".join(map(str, cfg.pi.default_resolution)))
        self.fps = self.number(cfg.pi.fps_min, cfg.pi.fps_max, cfg.pi.fps_default)
        self.bitrate = self.number(100, 25000, cfg.server.default_bitrate // 1000, step=100)
        settings.addRow("Résolution H.264", self.resolution)
        settings.addRow("Images/s demandées", self.fps)
        settings.addRow("Débit vidéo (kbit/s)", self.bitrate)
        layout.addLayout(settings)
        layout.addWidget(self.button("Appliquer au Pi", self.apply_camera, primary=True))
        layout.addWidget(self.note("L’application de ces trois réglages redémarre le flux vidéo. La reconnexion est automatique. Les réglages du Pi sont ceux de la session courante."))
        layout.addStretch()
        active = next(label for label, values in NETWORK_PRESETS.items() if values[0] == cfg.desktop.active_profile)
        self.network_profile.setCurrentText(active)

    def _ai_page(self):
        layout = self.page("IA")
        self.ai_enable = QCheckBox("Activer le traitement IA")
        self.ai_enable.setChecked(True)
        self.ai_enable.toggled.connect(self.enable_ai)
        layout.addWidget(self.ai_enable)
        models = list(dict.fromkeys([cfg.ai.default_model, "yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
                                    "yolo26l.pt", "yolo26x.pt", "yolo11n.pt", "yolo11s.pt",
                                    "yolov8n.pt", "sam2_t.pt", "sam3.pt"] +
                                   sorted(path.name for path in ROOT.glob("*.engine"))))
        self.model = self.combo(models, cfg.ai.default_model)
        self.model.setEditable(True)
        self.task = self.combo(["Detection", "Segmentation"])
        self.concepts = QLineEdit("person, cat, dog")
        self.confidence = self.number(0.1, 1., .5, decimals=2, step=.05)
        self.confidence.valueChanged.connect(core.update_conf)
        form = QFormLayout()
        form.addRow("Modèle", self.model)
        form.addRow("Tâche", self.task)
        form.addRow("Concepts SAM3", self.concepts)
        form.addRow("Confiance", self.confidence)
        layout.addLayout(form)
        layout.addWidget(self.button("Charger le modèle", self.load_model, primary=True))
        layout.addWidget(self.button("Choisir un fichier de modèle…", self.choose_model, background=False))
        layout.addWidget(self.button("Appliquer les concepts SAM3", self.apply_concepts))
        layout.addWidget(self.button("Optimiser avec TensorRT", lambda: self.run("Export TensorRT", core.ai.export_to_tensorrt)))
        layout.addWidget(self.note("Les modèles absents peuvent nécessiter un téléchargement. SAM3 et l’assistant visuel peuvent demander des droits d’accès Hugging Face. TensorRT nécessite le runtime NVIDIA compatible."))
        self.tracking = QCheckBox("Activer le suivi des objets")
        self.tracker = self.combo(["BoT-SORT", "ByteTrack"])
        layout.addWidget(self.tracking)
        layout.addWidget(self.tracker)
        layout.addWidget(self.button("Appliquer le suivi", self.apply_tracking))
        layout.addStretch()

    def _gallery_page(self, *, animal):
        layout = self.page("Animaux" if animal else "Visages")
        prefix = "animal" if animal else "face"
        enabled = QCheckBox("Activer la reconnaissance des animaux" if animal else "Activer la reconnaissance faciale")
        layout.addWidget(enabled)
        layout.addWidget(self.button("Appliquer la reconnaissance", lambda: self.toggle_recognition(animal), primary=True))
        name = QLineEdit()
        name.setPlaceholderText("Nom de l’animal" if animal else "Nom de la personne")
        threshold = self.number(.6 if animal else .2, .99 if animal else .8,
                                cfg.ai.animal_threshold if animal else cfg.ai.face_threshold,
                                decimals=2, step=.01)
        form = QFormLayout()
        form.addRow("Nom", name)
        form.addRow("Seuil de similarité", threshold)
        layout.addLayout(form)
        files = self.note("Aucune photo sélectionnée")
        layout.addWidget(files)
        layout.addWidget(self.button("Sélectionner des photos…", lambda: self.choose_photos(animal), background=False))
        layout.addWidget(self.button("Ajouter les photos à la galerie", lambda: self.import_photos(animal)))
        layout.addWidget(self.button("Capturer et apprendre · 10 vues", lambda: self.learn(animal), primary=True))
        layout.addWidget(self.button("Reconstruire la galerie", lambda: self.rebuild(animal)))
        status = self.note("Reconnaissance non chargée — active-la ou importe des photos pour commencer")
        layout.addWidget(status)
        layout.addWidget(self.note("Les photos et les galeries restent sur ce PC. L’apprentissage nécessite un sujet visible dans le flux et le modèle de reconnaissance installé."))
        layout.addStretch()
        for key, widget in (("enabled", enabled), ("name", name), ("threshold", threshold), ("files", files), ("status", status)):
            setattr(self, f"{prefix}_{key}", widget)

    def _alerts_page(self):
        layout = self.page("Alertes")
        self.email_enabled = QCheckBox("Activer les alertes email")
        layout.addWidget(self.email_enabled)
        self.email_sender = QLineEdit(core.email_config["sender"])
        self.email_password = QLineEdit()
        self.email_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.email_password.setPlaceholderText("Vide : conserver le mot de passe enregistré")
        self.email_receiver = QLineEdit(core.email_config["receiver"])
        form = QFormLayout()
        form.addRow("Adresse Gmail d’envoi", self.email_sender)
        form.addRow("Mot de passe d’application", self.email_password)
        form.addRow("Destinataire", self.email_receiver)
        layout.addLayout(form)
        layout.addWidget(self.button("Enregistrer les alertes", self.save_alerts, primary=True))
        layout.addWidget(self.button("Envoyer un email de test", lambda: self.run("Email de test", core.send_test_email)))
        self.email_status = self.note("Alertes désactivées au démarrage")
        layout.addWidget(self.email_status)
        layout.addWidget(self.note("Le mot de passe est chiffré localement. Les alertes automatiques incluent une image et la description du sujet détecté. Aucun email n’est envoyé avant activation ou clic sur le test."))
        layout.addStretch()

    def _assistant_page(self):
        layout = self.page("Assistant")
        layout.addWidget(self.note("Pose une question sur l’image courante. Le premier chargement du modèle peut prendre plusieurs minutes et nécessite un GPU CUDA ainsi que les poids du modèle."))
        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(False)
        layout.addWidget(self.chat, 1)
        self.question = QLineEdit()
        self.question.setPlaceholderText("Que vois-tu dans cette image ?")
        self.question.returnPressed.connect(self.ask)
        layout.addWidget(self.question)
        layout.addWidget(self.button("Analyser l’image", self.ask, primary=True))
        layout.addWidget(self.button("Effacer la conversation", self.clear_chat, background=False))

    def _logs_page(self):
        layout = self.page("Journal")
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(200)
        layout.addWidget(self.logs, 1)
        layout.addWidget(self.button("Ouvrir le dossier de l’application", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(ROOT))), background=False))
        layout.addWidget(self.note("Le flux est reçu directement par cette application. Aucun serveur web ni partage public n’est démarré."))

    def run(self, label, action, done=None):
        if self.worker.submit(label, action):
            self._completion = done
            self.statusBar().showMessage(label + "…")
            for button in self._actions:
                button.setEnabled(False)
        else:
            self.statusBar().showMessage("Une opération est déjà en cours — l’aperçu vidéo reste actif")

    def poll_commands(self):
        try:
            result = self.worker.results.get_nowait()
        except queue.Empty:
            if self._closing and not self.worker.busy:
                self.close()
            return
        done, self._completion = self._completion, None
        self.worker.acknowledge()
        for button in self._actions:
            button.setEnabled(True)
        if result.error:
            message = f"{result.label} : {result.error}"
            core.ui_log(message)
            self.statusBar().showMessage(message)
        else:
            if result.label == "Appareils Tailscale":
                message = f"{len(result.value)} appareil(s) détecté(s)"
            else:
                message = str(result.value) if result.value is not None else "Terminé"
            self.statusBar().showMessage(f"{result.label} : {message[:240]}")
            if result.label != "Analyse de l’image":
                core.ui_log(f"{result.label} : {message[:500]}")
            if done:
                done(result.value)
        if self._closing:
            self.close()

    def connect_camera(self):
        values = (self.host.text(), self.video_port.value(), self.ws_port.value(),
                  self.token.text(), self.open_timeout.value(), self.read_timeout.value())
        profile_id = NETWORK_PRESETS[self.network_profile.currentText()][0]

        def connect():
            message = save_connection(*values, profile_id=profile_id)
            core._RECONNECT.set()
            return message

        self.run("Connexion caméra", connect)

    def select_network_profile(self, profile):
        profile_id, opening, reading = NETWORK_PRESETS[profile]
        self.tailscale_button.setVisible(profile_id == "tailscale")
        saved = cfg.desktop.profiles.get(profile_id)
        if saved:
            self.host.setText(saved["host"])
            self.video_port.setValue(saved.get("video_port", 8888))
            self.ws_port.setValue(saved.get("ws_port", 8889))
            self.open_timeout.setValue(saved.get("open_timeout_ms", opening))
            self.read_timeout.setValue(saved.get("read_timeout_ms", reading))
        elif profile_id == "current":
            self.host.setText(cfg.pi.host)
            self.video_port.setValue(cfg.pi.video_port)
            self.ws_port.setValue(cfg.pi.ws_port)
            self.open_timeout.setValue(cfg.stream.capture_open_timeout_ms)
            self.read_timeout.setValue(cfg.stream.capture_read_timeout_ms)
        else:
            self.host.clear()
            self.host.setPlaceholderText("Adresse IPv4 ou nom MagicDNS" if profile_id == "tailscale" else "Adresse de la caméra pour ce profil")
            self.video_port.setValue(8888)
            self.ws_port.setValue(8889)
            self.open_timeout.setValue(opening)
            self.read_timeout.setValue(reading)
        if profile_id == "tailscale":
            self.statusBar().showMessage("Tailscale doit être actif sur ce PC et la caméra. Les adresses restent dans config.yaml, hors Git.")

    def detect_tailscale(self):
        def choose(devices):
            if self._closing:
                return
            if not devices:
                self.statusBar().showMessage("Aucun appareil Tailscale trouvé. Tu peux saisir l’adresse manuellement.")
                return
            labels = [f"{device['name']} — {device['host']} — {'en ligne' if device['online'] else 'hors ligne'}" for device in devices]
            label, accepted = QInputDialog.getItem(self, "Appareils Tailscale — données locales", "Choisir la caméra", labels, 0, False)
            if accepted:
                self.host.setText(devices[labels.index(label)]["host"])
                self.statusBar().showMessage("Adresse sélectionnée. Clique sur Enregistrer et reconnecter pour l’utiliser.")

        self.run("Appareils Tailscale", tailscale_devices, choose)

    def settings_received(self, values):
        self.resolution.setCurrentText(f"{values['width']}x{values['height']}")
        self.fps.setValue(values["fps"])
        self.bitrate.setValue(values["bitrate"] // 1000)
        self.statusBar().showMessage("Réglages confirmés par le Pi")

    def read_settings(self):
        self.run("Lecture des réglages", camera_settings, self.settings_received)

    def apply_camera(self):
        width, height = map(int, self.resolution.currentText().split("x"))
        values = {"width": width, "height": height, "fps": self.fps.value(), "bitrate": self.bitrate.value() * 1000}
        self.run("Réglages caméra", lambda: camera_settings(values), self.settings_received)

    def enable_ai(self, enabled):
        with core.state.lock:
            core.state.ai_enabled = enabled

    def load_model(self):
        values = (self.model.currentText(), self.task.currentText(), self.concepts.text())
        self.run("Chargement du modèle", lambda: core.change_model_logic(*values))

    def choose_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Modèle IA", str(ROOT), "Modèles (*.pt *.engine)")
        if filename:
            path = Path(filename)
            self.model.setCurrentText(path.name if path.parent == ROOT else str(path))

    def apply_concepts(self):
        concepts = self.concepts.text()
        self.run("Concepts SAM3", lambda: core.update_sam3_prompt_logic(concepts))

    def apply_tracking(self):
        enabled, tracker = self.tracking.isChecked(), self.tracker.currentText()

        def apply():
            core.change_tracker(tracker)
            core.toggle_tracking(enabled)
            return f"{tracker} — {'actif' if enabled else 'désactivé'}"

        self.run("Suivi des objets", apply)

    def toggle_recognition(self, animal):
        enabled = (self.animal_enabled if animal else self.face_enabled).isChecked()
        threshold = (self.animal_threshold if animal else self.face_threshold).value()
        action = core.toggle_animal_recognition if animal else core.toggle_face_recognition
        set_threshold = core.update_animal_sensitivity if animal else core.update_sensitivity
        target = self.animal_status if animal else self.face_status

        def apply():
            message = action(enabled)
            if enabled:
                set_threshold(threshold)
            return message

        self.run("Reconnaissance", apply, target.setText)

    def choose_photos(self, animal):
        files, _ = QFileDialog.getOpenFileNames(self, "Photos pour la galerie", str(ROOT),
                                               "Photos (*.jpg *.jpeg *.png *.webp)")
        if files:
            if animal:
                self._animal_files = files
                self.animal_files.setText(f"{len(files)} photo(s) sélectionnée(s)")
            else:
                self._face_files = files
                self.face_files.setText(f"{len(files)} photo(s) sélectionnée(s)")

    def import_photos(self, animal):
        name = (self.animal_name if animal else self.face_name).text()
        files = list(self._animal_files if animal else self._face_files)
        action = core.add_animal_to_gallery if animal else core.add_to_gallery
        target = self.animal_status if animal else self.face_status
        self.run("Import des photos", lambda: action(name, files), target.setText)

    def rebuild(self, animal):
        action = core.rebuild_animal_gallery if animal else core.rebuild_gallery
        target = self.animal_status if animal else self.face_status
        self.run("Reconstruction de la galerie", action, target.setText)

    def learn(self, animal):
        name = (self.animal_name if animal else self.face_name).text()
        action = core.start_animal_learning if animal else core.start_learning
        target = self.animal_status if animal else self.face_status
        self.run("Apprentissage", lambda: action(name), target.setText)

    def save_alerts(self):
        values = (self.email_enabled.isChecked(), self.email_sender.text(),
                  self.email_password.text(), self.email_receiver.text())

        def saved(message):
            self.email_password.clear()
            self.email_status.setText(message)

        self.run("Enregistrement des alertes", lambda: core.save_email_settings(*values), saved)

    def ask(self):
        question = self.question.text().strip()
        if not question or self.worker.busy:
            return
        history = list(self._history)
        self.chat.append("Vous : " + html.escape(question))
        self.question.clear()

        def answered(answer):
            self.chat.append("Assistant : " + html.escape(str(answer)))
            self._history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": str(answer)}])
            self._history = self._history[-20:]

        self.run("Analyse de l’image", lambda: core.ask_vlm(question, history), answered)

    def clear_chat(self):
        if not self.worker.busy:
            self._history.clear()
            self.chat.clear()

    def refresh(self):
        now = time.monotonic()
        with core.state.lock:
            state = core.state
            age = now - state.last_frame_time if state.last_frame_time else float("inf")
            annotated = (self.overlay.isChecked() and state.ai_enabled and
                         state.last_processed is not None and now - state.last_processed_time < 1)
            frame = state.last_processed if annotated else state.last_frame
            key = (annotated, state.processed_id if annotated else state.frame_id)
            status, info = state.connection_status, state.last_info
            times = tuple(state.frame_times)
        self.connection.setText("Connectée" if age < 3 else status if age == float("inf") else "Flux interrompu")
        waiting = "Flux interrompu — attente de données…" if status == "Connectée" else status
        notice = "Affichage en pause" if self.pause.isChecked() else waiting if age > 3 else ""
        repaint = notice != self.video.notice
        self.video.notice = notice
        if not self.pause.isChecked() and frame is not None and key != self._last_image_key:
            self.video.image = frame_image(frame)
            self._snapshot = frame
            self._last_image_key = key
            self._painted += 1
            repaint = True
        if repaint:
            self.video.update()
        self.snapshot_button.setEnabled(self._snapshot is not None)
        if now - self._metric_time >= 1:
            recent = [stamp for stamp in times if now - stamp <= 2]
            received = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0
            displayed = self._painted / (now - self._metric_time)
            shape = f"{frame.shape[1]} × {frame.shape[0]}" if frame is not None else "—"
            self.metrics.setText(f"{shape}   ·   Réception {received:.1f} i/s   ·   Affichage {displayed:.1f} i/s")
            if core.ai._loader_error:
                info = "IA indisponible : " + core.ai._loader_error
            self.ai_status.setText(f"{core.ai.segment_model_name} · {core.ai.device} · {info}"
                                   if core.state.ai_enabled else "Traitement IA désactivé — aperçu direct")
            self._painted = 0
            self._metric_time = now
            logs = core.get_logs()
            if logs != self._last_log:
                self.logs.setPlainText(logs)
                self._last_log = logs

    def save_snapshot(self):
        if self._snapshot is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Enregistrer la capture", f"camera-{time.strftime('%Y%m%d-%H%M%S')}.png", "Image PNG (*.png)")
        if filename:
            frame = self._snapshot.copy()

            def save():
                ok, encoded = cv2.imencode(".png", frame)
                if not ok:
                    raise ValueError("Impossible d’encoder la capture")
                Path(filename).write_bytes(encoded.tobytes())
                return filename

            self.run("Capture de l’image", save)

    def fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def closeEvent(self, event):
        if self.worker.busy:
            self._closing = True
            self.statusBar().showMessage("Fermeture après la fin de l’opération en cours…")
            event.ignore()
            return
        core._STOP.set()
        if any(worker.is_alive() for worker in core._background_threads):
            self._closing = True
            self.statusBar().showMessage("Fermeture — libération de la caméra et du moteur IA…")
            event.ignore()
            return
        self.preferences.setValue("geometry", self.saveGeometry())
        self.timer.stop()
        self.slow_timer.stop()
        self.worker.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Smart Camera — application Windows native")
    parser.add_argument("--no-camera", action="store_true", help="Ouvrir l’interface sans connexion ni chargement IA")
    args = parser.parse_args()
    file_handler = logging.FileHandler(ROOT / "desktop.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    apply_windows_boost()
    application = QApplication(sys.argv[:1])
    application.setApplicationName("Smart Camera")
    application.setOrganizationName("CameraAI")
    window = CameraWindow(start_camera=not args.no_camera)
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
