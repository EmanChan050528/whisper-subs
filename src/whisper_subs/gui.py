"""The drag-and-drop window: `whisper-subs-gui`.

Drop files in, press Start, get subtitles beside them. Everything runs through
job.run_job on a worker thread; the window only shows what it reports. Stop is
a pause: each stage checkpoints, so pressing Start again carries on.
"""

import os
import sys
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from whisper_subs import __version__, glossary, ollama
from whisper_subs.job import JobError, Settings, Stopped, run_job
from whisper_subs.transcribe import DEFAULT_MODEL, FAST_MODEL

MEDIA = ("*.mp4 *.mkv *.webm *.mov *.avi *.flv *.ts *.m4v "
         "*.mp3 *.m4a *.aac *.wav *.flac *.ogg *.opus")
WHISPER_CHOICES = [(f"{DEFAULT_MODEL}  (accurate)", DEFAULT_MODEL),
                   (f"{FAST_MODEL}  (about 4x faster)", FAST_MODEL)]

WAITING, RUNNING, DONE, PAUSED, FAILED = "Waiting", "Running", "Done", "Paused", "Error"
STATUS_COLOURS = {DONE: "#2e7d32", PAUSED: "#b26a00", FAILED: "#c62828"}


class Worker(QObject):
    """Runs queued files one after another on its own thread."""

    log = Signal(str)
    progress = Signal(str, float, float)
    started_file = Signal(int)
    finished_file = Signal(int, str, str, list)  # row, status, detail, outputs
    finished = Signal()

    def __init__(self, jobs: list[tuple[int, Path]], settings: Settings,
                 stop: threading.Event):
        super().__init__()
        self.jobs, self.settings, self.stop = jobs, settings, stop

    def run(self) -> None:
        for row, path in self.jobs:
            if self.stop.is_set():
                break
            self.started_file.emit(row)
            self.log.emit(f"— {path.name}")
            try:
                result = run_job(path, self.settings, log=self.log.emit,
                                 progress=self.progress.emit, should_stop=self.stop.is_set)
            except Stopped:
                self.finished_file.emit(row, PAUSED, "press Start to carry on", [])
                break
            except JobError as err:
                self.log.emit(f"error: {err}")
                self.finished_file.emit(row, FAILED, str(err), [])
                continue
            except Exception as err:  # keep the window alive; show what happened
                self.log.emit(traceback.format_exc())
                self.finished_file.emit(row, FAILED, f"{type(err).__name__}: {err}", [])
                continue
            detail = (f"{result.missing} line(s) untranslated, Start again to retry them"
                      if result.missing else "")
            self.finished_file.emit(row, DONE, detail, [str(p) for p in result.outputs])
        self.finished.emit()


class DropZone(QFrame):
    """The big target in the middle. Also clickable, for people who don't drag."""

    dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("drop")
        self.setMinimumHeight(90)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        label = QLabel("Drop video or audio files here\nor click to choose")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setObjectName("dropLabel")
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        self._set_hover(False)

    def _set_hover(self, on: bool) -> None:
        self.setProperty("hover", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose files to subtitle", "",
            f"Video or audio ({MEDIA});;Transcripts (*.ja.json);;All files (*)")
        if files:
            self.dropped.emit([Path(f) for f in files])

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_hover(True)

    def dragLeaveEvent(self, event) -> None:
        self._set_hover(False)

    def dropEvent(self, event: QDropEvent) -> None:
        self._set_hover(False)
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        files = []
        for p in paths:  # a dropped folder contributes the media files inside it
            if p.is_dir():
                files += sorted(f for f in p.iterdir() if _is_media(f))
            elif p.is_file():
                files.append(p)
        if files:
            self.dropped.emit(files)


def _is_media(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".ja.json") or
                               f"*{path.suffix.lower()}" in MEDIA.split())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Subtitler")
        self.resize(760, 720)
        # An INI beside the glossaries, not the registry: easy to find, and
        # WHISPER_SUBS_HOME moves it (which is what keeps tests off real settings).
        self.prefs = QSettings(str(glossary.home() / "gui.ini"), QSettings.Format.IniFormat)
        self.stop = threading.Event()
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self.last_output: Path | None = None

        self.drop = DropZone()
        self.drop.dropped.connect(self.add_files)

        self.queue = QListWidget()
        self.queue.setAlternatingRowColors(True)
        self.queue.setMinimumHeight(90)
        self.queue.setMaximumHeight(130)  # the log gets the spare height

        # Settings
        self.whisper = QComboBox()
        for label, value in WHISPER_CHOICES:
            self.whisper.addItem(label, value)
        self.llm = QComboBox()
        self.llm.setEditable(True)
        self.glossary = QComboBox()
        self.glossary.setEditable(True)
        self.glossary.lineEdit().setPlaceholderText("none — or type a series name")
        glossary_folder = QPushButton("Edit…")
        glossary_folder.setToolTip("Open the glossary folder. Fix a name in its .json file and "
                                   "the correction is kept for every later file.")
        glossary_folder.clicked.connect(self.open_glossary_folder)
        glossary_row = QHBoxLayout()
        glossary_row.addWidget(self.glossary, 1)
        glossary_row.addWidget(glossary_folder)
        self.bilingual = QCheckBox("Also write a bilingual file (Japanese above English)")
        self.ja_only = QCheckBox("Japanese only (skip translation; Ollama not needed)")

        form = QFormLayout()
        form.addRow("Transcription model", self.whisper)
        form.addRow("Translation model", self.llm)
        form.addRow("Series glossary", glossary_row)
        form.addRow("", self.bilingual)
        form.addRow("", self.ja_only)
        box = QGroupBox("Settings")
        box.setLayout(form)

        self.health = QLabel()
        self.health.setWordWrap(True)
        self.health.setObjectName("health")

        # Controls and progress
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Pauses after the current step. Start carries on from there.")
        self.stop_btn.clicked.connect(self.request_stop)
        self.clear_btn = QPushButton("Clear finished")
        self.clear_btn.clicked.connect(self.clear_finished)
        self.open_btn = QPushButton("Open output folder")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_output)
        buttons = QHBoxLayout()
        for b in (self.start_btn, self.stop_btn, self.clear_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.open_btn)

        self.status = QLabel("Add some files to begin.")
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setMaximumHeight(8)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.log.setFont(mono)
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        root = QVBoxLayout()
        root.addWidget(self.drop)
        root.addWidget(self.queue)
        root.addWidget(box)
        root.addWidget(self.health)
        root.addLayout(buttons)
        root.addWidget(self.status)
        root.addWidget(self.bar)
        root.addWidget(self.log, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self.setStyleSheet(STYLE)

        self.load_prefs()
        self.check_health()
        self.ja_only.toggled.connect(self.update_enabled)
        self.update_enabled()

    # ------------------------------------------------------------ settings

    def load_prefs(self) -> None:
        models = ollama.list_models()
        self._ollama_models = models
        wanted = self.prefs.value("llm_model", ollama.DEFAULT_MODEL)
        for name in models or []:
            self.llm.addItem(name)
        if self.llm.findText(wanted) < 0:
            self.llm.addItem(wanted)
        self.llm.setCurrentText(wanted)
        self.whisper.setCurrentIndex(max(0, self.whisper.findData(
            self.prefs.value("whisper_model", DEFAULT_MODEL))))
        gdir = glossary.home() / "glossaries"
        self.glossary.addItem("")
        for g in sorted(gdir.glob("*.json")) if gdir.is_dir() else []:
            self.glossary.addItem(g.stem)
        self.glossary.setCurrentText(self.prefs.value("glossary", ""))
        self.bilingual.setChecked(self.prefs.value("bilingual", False, type=bool))
        self.ja_only.setChecked(self.prefs.value("ja_only", False, type=bool))

    def save_prefs(self) -> None:
        self.prefs.setValue("llm_model", self.llm.currentText().strip())
        self.prefs.setValue("whisper_model", self.whisper.currentData())
        self.prefs.setValue("glossary", self.glossary.currentText().strip())
        self.prefs.setValue("bilingual", self.bilingual.isChecked())
        self.prefs.setValue("ja_only", self.ja_only.isChecked())

    def settings(self) -> Settings:
        g = self.glossary.currentText().strip() or None
        return Settings(model=self.whisper.currentData(), glossary=g,
                        llm_model=self.llm.currentText().strip() or ollama.DEFAULT_MODEL,
                        bilingual=self.bilingual.isChecked(), ja_only=self.ja_only.isChecked())

    def check_health(self) -> None:
        """What will and won't work, before anyone presses Start."""
        notes = []
        try:
            from whisper_subs.gpu import enable_cuda_dlls
            enable_cuda_dlls()
            import ctranslate2
            gpu = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            gpu = False
        notes.append("GPU: NVIDIA CUDA ready." if gpu else
                     "GPU: none found. Transcription will run on the CPU and be slow; "
                     "the faster model helps.")
        if self._ollama_models is None:
            notes.append("Ollama: not running. Start the Ollama app to translate, or tick "
                         "Japanese only.")
        elif not self._ollama_models:
            notes.append(f"Ollama: running, but no models. Run: ollama pull "
                         f"{ollama.DEFAULT_MODEL}")
        else:
            notes.append(f"Ollama: running, {len(self._ollama_models)} model(s).")
        self.health.setText("   ".join(notes))

    def update_enabled(self) -> None:
        on = not self.ja_only.isChecked()
        for w in (self.llm, self.bilingual, self.glossary):
            w.setEnabled(on)

    def open_glossary_folder(self) -> None:
        folder = glossary.home() / "glossaries"
        folder.mkdir(parents=True, exist_ok=True)
        _open(folder)

    # --------------------------------------------------------------- queue

    def add_files(self, paths: list[Path]) -> None:
        existing = {self.queue.item(i).data(Qt.ItemDataRole.UserRole)
                    for i in range(self.queue.count())}
        added = 0
        for p in paths:
            if str(p) in existing:
                continue
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self._set_row(item, WAITING)
            self.queue.addItem(item)
            added += 1
        if added and not self.running:
            self.status.setText(f"{self._count(WAITING)} file(s) ready. Press Start.")

    def _set_row(self, item: QListWidgetItem, status: str, detail: str = "") -> None:
        name = Path(item.data(Qt.ItemDataRole.UserRole)).name
        item.setData(Qt.ItemDataRole.UserRole + 1, status)
        item.setText(f"{status:<10}{name}" + (f"   — {detail}" if detail else ""))
        item.setForeground(QColor(STATUS_COLOURS.get(status, "#222222")))

    def _count(self, status: str) -> int:
        return sum(self.queue.item(i).data(Qt.ItemDataRole.UserRole + 1) == status
                   for i in range(self.queue.count()))

    def clear_finished(self) -> None:
        for i in reversed(range(self.queue.count())):
            if self.queue.item(i).data(Qt.ItemDataRole.UserRole + 1) == DONE:
                self.queue.takeItem(i)

    # ----------------------------------------------------------------- run

    @property
    def running(self) -> bool:
        return self.thread is not None

    def start(self) -> None:
        jobs = [(i, Path(self.queue.item(i).data(Qt.ItemDataRole.UserRole)))
                for i in range(self.queue.count())
                if self.queue.item(i).data(Qt.ItemDataRole.UserRole + 1) != DONE]
        if not jobs:
            self.status.setText("Nothing to do. Drop some files in first.")
            return
        self.save_prefs()
        self.stop.clear()
        self.thread = QThread()
        self.worker = Worker(jobs, self.settings(), self.stop)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.on_progress)
        self.worker.started_file.connect(self.on_started)
        self.worker.finished_file.connect(self.on_finished_file)
        self.worker.finished.connect(self.on_all_done)
        self.set_running(True)
        self.thread.start()

    def request_stop(self) -> None:
        self.stop.set()
        self.stop_btn.setEnabled(False)
        self.status.setText("Stopping after the current step…")

    def set_running(self, on: bool) -> None:
        self.start_btn.setEnabled(not on)
        self.stop_btn.setEnabled(on)
        self.drop.setEnabled(True)  # adding more files while running is fine
        for w in (self.whisper, self.llm, self.glossary, self.bilingual, self.ja_only):
            w.setEnabled(not on)
        if not on:
            self.update_enabled()

    def on_started(self, row: int) -> None:
        self.current = row
        self._set_row(self.queue.item(row), RUNNING)
        self.bar.setRange(0, 0)  # busy until the first progress report
        name = Path(self.queue.item(row).data(Qt.ItemDataRole.UserRole)).name
        self.status.setText(f"Working on {name}")

    def on_progress(self, stage: str, done: float, total: float) -> None:
        self.bar.setRange(0, 1000)
        self.bar.setValue(int(1000 * done / total) if total else 0)
        item = self.queue.item(self.current)
        if stage == "transcribe":
            text = f"transcribing {100 * done / total:.0f}%" if total else "transcribing"
        else:
            text = f"translating {int(done)}/{int(total)}"
        self._set_row(item, RUNNING, text)
        self.status.setText(f"{Path(item.data(Qt.ItemDataRole.UserRole)).name}: {text}")

    def on_finished_file(self, row: int, status: str, detail: str, outputs: list) -> None:
        self._set_row(self.queue.item(row), status, detail)
        if outputs:
            self.last_output = Path(outputs[0]).parent
            self.open_btn.setEnabled(True)
            for o in outputs:
                self.append_log(f"  wrote {o}")

    def on_all_done(self) -> None:
        self.thread.quit()
        self.thread.wait()
        self.thread = self.worker = None
        self.set_running(False)
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        left = self._count(WAITING) + self._count(PAUSED)
        failed = self._count(FAILED)
        if self.stop.is_set():
            self.status.setText("Paused. Press Start to carry on where it stopped.")
        elif failed:
            self.status.setText(f"Finished, with {failed} error(s): see the list and the log.")
        else:
            self.status.setText("All done." if not left else f"{left} file(s) still waiting.")

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text)
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())  # follow the newest line

    def open_output(self) -> None:
        if self.last_output:
            _open(self.last_output)

    def closeEvent(self, event) -> None:
        self.save_prefs()
        if self.running:
            self.stop.set()
            self.thread.quit()
            self.thread.wait(3000)
        event.accept()


def _open(folder: Path) -> None:
    if sys.platform == "win32":
        os.startfile(folder)
    else:
        import subprocess
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(folder)])


STYLE = """
QMainWindow { background: #fafafa; }
#drop { border: 2px dashed #9e9e9e; border-radius: 10px; background: #ffffff; }
#drop[hover="true"] { border-color: #1565c0; background: #e3f2fd; }
#dropLabel { color: #555555; font-size: 15px; }
#health { color: #555555; }
QPushButton { padding: 6px 14px; }
QPushButton#primary { background: #1565c0; color: white; border: none; border-radius: 4px;
                      font-weight: 600; }
QPushButton#primary:disabled { background: #90a4ae; }
QProgressBar { border: none; background: #e0e0e0; border-radius: 4px; }
QProgressBar::chunk { background: #1565c0; border-radius: 4px; }
QListWidget { font-family: Consolas, monospace; }
"""


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Whisper Subtitler")
    app.setApplicationVersion(__version__)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
