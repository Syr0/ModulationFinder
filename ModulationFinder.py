import sys
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.stats import kurtosis
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog,
                             QVBoxLayout, QHBoxLayout, QWidget, QSplitter,
                             QLabel, QSpinBox, QListWidget, QAbstractItemView,
                             QPushButton, QGraphicsRectItem, QSlider, QProgressBar,
                             QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QBrush
import pyqtgraph as pg


class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.setMenuEnabled(False)
        self.setMouseMode(self.PanMode)
        self.custom_rect = None

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self.autoRange()
            ev.accept()
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.RightButton:
            ev.accept()
            pos = ev.pos()
            start_pos = ev.buttonDownPos()

            rect_view = pg.QtCore.QRectF(start_pos, pos).normalized()
            rect_data = self.childGroup.mapRectFromParent(rect_view)

            if ev.isStart():
                if not self.custom_rect:
                    self.custom_rect = QGraphicsRectItem()
                    self.custom_rect.setPen(pg.mkPen((255, 255, 255), width=1, style=Qt.DashLine))
                    self.custom_rect.setBrush(pg.mkBrush(None))
                    self.custom_rect.setZValue(1e9)
                    self.addItem(self.custom_rect, ignoreBounds=True)
                self.custom_rect.setRect(rect_data)
                self.custom_rect.show()
            elif ev.isFinish():
                if self.custom_rect:
                    self.custom_rect.hide()
                self.showAxRect(rect_data)
                self.axHistoryPointer += 1
                self.axHistory = self.axHistory[:self.axHistoryPointer] + [rect_data]
            else:
                if self.custom_rect:
                    self.custom_rect.setRect(rect_data)
        else:
            super().mouseDragEvent(ev, axis)


class HoverRectItem(QGraphicsRectItem):
    def __init__(self, x, y, w, h, text_info, parent=None):
        super().__init__(x, y, w, h, parent)
        self.setAcceptHoverEvents(True)

        self.normal_pen = pg.mkPen((0, 255, 255), width=2)
        self.hover_pen = pg.mkPen((0, 255, 255), width=3)

        self.normal_brush = QBrush(QColor(0, 0, 0, 0))
        self.hover_brush = QBrush(QColor(0, 255, 255, 40))

        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)
        self.setToolTip(text_info)
        self.setZValue(1)

    def hoverEnterEvent(self, event):
        self.setPen(self.hover_pen)
        self.setBrush(self.hover_brush)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)
        super().hoverLeaveEvent(event)


class AudioLoader(QThread):
    finished = pyqtSignal(np.ndarray, int)
    error = pyqtSignal()

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            data, samplerate = sf.read(self.filepath, always_2d=True)
            self.finished.emit(data[:, 0].astype(np.float32), samplerate)
        except Exception:
            self.error.emit()


class SpectrogramWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)

    def __init__(self, data, fs):
        super().__init__()
        self.data = data
        self.fs = fs

    def run(self):
        sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
        results = {}
        total_steps = len(sizes)
        for i, nperseg in enumerate(sizes):
            noverlap = int(nperseg * 0.75)
            f, t, Sxx = signal.spectrogram(self.data, fs=self.fs, window='hamming',
                                           nperseg=nperseg, noverlap=noverlap, mode='magnitude')
            Sxx_log = 10 * np.log10(Sxx + 1e-10)
            results[nperseg] = {
                'f': f,
                't': t,
                'Sxx_log': Sxx_log
            }
            self.progress.emit(int(((i + 1) / total_steps) * 100))
        self.finished.emit(results)


class BruteForceScanner(QThread):
    finished = pyqtSignal(list)
    progress = pyqtSignal(int)

    def __init__(self, data, fs, Sxx_log, f_axis, s_min, s_max, num_freqs, modulations, top_percent, equidistant_mode):
        super().__init__()
        self.data = data
        self.fs = fs
        self.Sxx_log = Sxx_log
        self.f_axis = f_axis
        self.s_min = s_min
        self.s_max = s_max
        self.num_freqs = num_freqs
        self.modulations = modulations
        self.top_percent = top_percent
        self.equidistant_mode = equidistant_mode

    def run(self):
        raw_detections = []
        if len(self.data) == 0 or len(self.modulations) == 0 or self.Sxx_log is None:
            self.finished.emit(raw_detections)
            return

        mean_power = np.mean(self.Sxx_log, axis=1)
        peaks, _ = signal.find_peaks(mean_power, distance=len(self.f_axis) // 100)
        if len(peaks) == 0:
            self.finished.emit(raw_detections)
            return

        peak_freqs = self.f_axis[peaks[np.argsort(mean_power[peaks])[-self.num_freqs * 3:]]]
        peak_freqs.sort()

        if self.equidistant_mode and self.num_freqs > 1:
            raw_detections = self._scan_equidistant(peak_freqs)
        else:
            raw_detections = self._scan_independent(peak_freqs)

        raw_detections.sort(key=lambda x: (x['duration'], x['guete']), reverse=True)
        suppressed = []
        for det in raw_detections:
            overlap = False
            for s_det in suppressed:
                t_overlap = max(0, min(det['t_start'] + det['duration'], s_det['t_start'] + s_det['duration']) - max(
                    det['t_start'], s_det['t_start'])) > 0
                f_overlap = abs(det['f_center'] - s_det['f_center']) < (det['bandwidth'] / 2)
                if t_overlap and f_overlap:
                    overlap = True
                    break
            if not overlap:
                suppressed.append(det)

        if len(suppressed) > 0:
            num_to_keep = max(1, int(len(suppressed) * (self.top_percent / 100.0)))
            final_detections = suppressed[:num_to_keep]
        else:
            final_detections = []

        self.finished.emit(final_detections)

    def _estimate_baudrate(self, baseband_signal):
        power_env = np.abs(baseband_signal) ** 2
        power_env -= np.mean(power_env)
        fft_size = int(2 ** np.ceil(np.log2(len(power_env))))
        fft_val = np.abs(np.fft.rfft(power_env, n=fft_size))
        fftfreqs = np.fft.rfftfreq(fft_size, d=1.0 / self.fs)

        valid_bins = (fftfreqs >= self.s_min) & (fftfreqs <= self.s_max)
        if not np.any(valid_bins):
            return None

        valid_fft = fft_val[valid_bins]
        valid_freqs = fftfreqs[valid_bins]
        peak_idx = np.argmax(valid_fft)
        return valid_freqs[peak_idx]

    def _extract_active_chunks(self, channel_bb):
        power_env = np.abs(channel_bb)
        b_env, a_env = signal.butter(2, 50 / (self.fs / 2), btype='low')
        smoothed_env = signal.filtfilt(b_env, a_env, power_env)

        max_env = np.max(smoothed_env)
        if max_env < 1e-6: return []

        threshold = max_env * 0.15
        is_active = smoothed_env > threshold

        diff = np.diff(is_active.astype(int))
        starts = np.where(diff == 1)[0]
        stops = np.where(diff == -1)[0]

        if is_active[0]: starts = np.insert(starts, 0, 0)
        if is_active[-1]: stops = np.append(stops, len(is_active) - 1)

        chunks = []
        for start_idx, stop_idx in zip(starts, stops):
            duration = (stop_idx - start_idx) / self.fs
            if duration >= 0.1:
                chunks.append((start_idx, stop_idx, duration))
        return chunks

    def _evaluate_modulation(self, chunk, speed):
        t_chunk = np.arange(len(chunk)) / self.fs
        chunk_norm = chunk / (np.max(np.abs(chunk)) + 1e-9)
        window = signal.windows.hann(len(chunk_norm))

        offsets = {}
        for M_val in [2, 4, 8]:
            v = (chunk_norm ** M_val) * window
            fft_v = np.fft.fft(v)
            fftfreqs = np.fft.fftfreq(len(v), d=1.0 / self.fs)

            valid_bins = (np.abs(fftfreqs) > 10) & (np.abs(fftfreqs) < self.s_max * M_val)
            abs_fft = np.abs(fft_v)
            abs_fft[~valid_bins] = 0

            if np.any(abs_fft > 0):
                peak_idx = np.argmax(abs_fft)
                offsets[M_val] = fftfreqs[peak_idx] / M_val
            else:
                offsets[M_val] = 0.0

        sps = int(self.fs / speed)
        if sps < 2: return 0.0, "", 0.0

        b_sym, a_sym = signal.butter(3, (speed * 0.6) / (self.fs / 2), btype='low')
        filtered_chunks = {}
        for M_val in [2, 4, 8]:
            f_offset = offsets[M_val]
            shifted = chunk * np.exp(-1j * 2 * np.pi * f_offset * t_chunk)
            filtered_chunks[M_val] = signal.filtfilt(b_sym, a_sym, shifted)

        mods_by_M = {
            2: [m for m in self.modulations if m == "2-PSK"],
            4: [m for m in self.modulations if m in ["4-PSK", "16-QAM", "64-QAM"]],
            8: [m for m in self.modulations if m == "8-PSK"]
        }

        current_speed_results = []
        for M_val, mods in mods_by_M.items():
            if not mods: continue
            filtered = filtered_chunks[M_val]

            for phase in range(0, sps, max(1, sps // 3)):
                symbols = filtered[phase::sps]
                if len(symbols) < 20: continue

                mean_amp = np.mean(np.abs(symbols))
                symbols = symbols / (mean_amp + 1e-9)

                k_val = kurtosis(np.real(symbols)) + kurtosis(np.imag(symbols))
                if not np.isnan(k_val) and k_val > -0.1: continue

                cv_amp = np.var(np.abs(symbols)) / (np.mean(np.abs(symbols)) ** 2 + 1e-9)
                is_qam_signal = cv_amp > 0.15

                for mod_name in mods:
                    if "QAM" in mod_name and not is_qam_signal: continue
                    if "PSK" in mod_name and is_qam_signal: continue

                    if mod_name == "2-PSK":
                        exp_peak = 1.0
                    elif mod_name == "4-PSK":
                        exp_peak = 1.0
                    elif mod_name == "8-PSK":
                        exp_peak = 1.0
                    elif mod_name == "16-QAM":
                        exp_peak = 0.68
                    elif mod_name == "64-QAM":
                        exp_peak = 0.61

                    v_sym = symbols ** M_val
                    peak = np.max(np.abs(np.fft.fft(v_sym))) / len(v_sym)
                    guete = (peak / exp_peak) * 100
                    current_speed_results.append((guete, mod_name, offsets[M_val]))

        if not current_speed_results: return 0.0, "", 0.0
        current_speed_results.sort(key=lambda x: x[0], reverse=True)

        psk2_res = next((r for r in current_speed_results if r[1] == "2-PSK"), None)
        psk4_res = next((r for r in current_speed_results if r[1] == "4-PSK"), None)

        if psk2_res and psk2_res[0] > 70:
            best_res = psk2_res
        elif psk4_res and psk4_res[0] > 70:
            best_res = psk4_res
        else:
            best_res = current_speed_results[0]

        return best_res[0], best_res[1], best_res[2]

    def _scan_equidistant(self, peak_freqs):
        best_sum_guete = 0
        best_candidate_set = []
        t_vector_full = np.arange(len(self.data)) / self.fs
        f_tol = (self.f_axis[1] - self.f_axis[0]) * 2

        candidates = []
        for i in range(len(peak_freqs)):
            for j in range(i + 1, len(peak_freqs)):
                df = peak_freqs[j] - peak_freqs[i]
                f0 = peak_freqs[i]
                expected_freqs = [f0 + k * df for k in range(self.num_freqs)]
                matches = sum(1 for ef in expected_freqs if any(abs(ef - pf) < f_tol for pf in peak_freqs))

                if matches >= self.num_freqs * 0.7:
                    candidates.append(expected_freqs)

        total_iters = len(candidates)
        for idx, channel_set in enumerate(candidates):
            current_sum_guete = 0
            current_detections = []

            center_freq = channel_set[len(channel_set) // 2]
            bb_center = self.data * np.exp(-1j * 2 * np.pi * center_freq * t_vector_full)
            b_chan, a_chan = signal.butter(3, (self.s_max / 2 + 500) / (self.fs / 2), btype='low')
            bb_center = signal.filtfilt(b_chan, a_chan, bb_center)

            shared_baudrate = self._estimate_baudrate(bb_center)
            if not shared_baudrate: continue

            active_chunks = self._extract_active_chunks(bb_center)
            if not active_chunks: continue

            for start_idx, stop_idx, duration in active_chunks:
                t_start = start_idx / self.fs

                for fc in channel_set:
                    bb_chan = self.data[start_idx:stop_idx] * np.exp(
                        -1j * 2 * np.pi * fc * t_vector_full[start_idx:stop_idx])
                    bb_chan = signal.filtfilt(b_chan, a_chan, bb_chan)

                    guete, mod, f_offset = self._evaluate_modulation(bb_chan, shared_baudrate)

                    if guete > 40:
                        current_sum_guete += guete
                        current_detections.append({
                            't_start': t_start,
                            'duration': duration,
                            'f_center': fc + f_offset,
                            'bandwidth': shared_baudrate * 1.2,
                            'mod': mod,
                            'speed': shared_baudrate,
                            'guete': min(100, guete)
                        })

            if current_sum_guete > best_sum_guete:
                best_sum_guete = current_sum_guete
                best_candidate_set = current_detections

            self.progress.emit(int(((idx + 1) / max(1, total_iters)) * 100))

        return best_candidate_set

    def _scan_independent(self, peak_freqs):
        raw_detections = []
        t_vector_full = np.arange(len(self.data)) / self.fs
        total_iters = len(peak_freqs)

        for idx, fc in enumerate(peak_freqs):
            baseband = self.data * np.exp(-1j * 2 * np.pi * fc * t_vector_full)
            b_chan, a_chan = signal.butter(3, (self.s_max / 2 + 500) / (self.fs / 2), btype='low')
            channel_bb = signal.filtfilt(b_chan, a_chan, baseband)

            active_chunks = self._extract_active_chunks(channel_bb)
            if not active_chunks: continue

            for start_idx, stop_idx, duration in active_chunks:
                chunk = channel_bb[start_idx:stop_idx]
                t_start = start_idx / self.fs

                baudrate = self._estimate_baudrate(chunk)
                if not baudrate: continue

                guete, mod, f_offset = self._evaluate_modulation(chunk, baudrate)

                if guete > 40:
                    raw_detections.append({
                        't_start': t_start,
                        'duration': duration,
                        'f_center': fc + f_offset,
                        'bandwidth': baudrate * 1.2,
                        'mod': mod,
                        'speed': baudrate,
                        'guete': min(100, guete)
                    })

            self.progress.emit(int(((idx + 1) / max(1, total_iters)) * 100))

        return raw_detections


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDR SigInt Scanner (M-th Power FFT)")
        self.resize(1600, 900)
        self.setAcceptDrops(True)
        self.audio_data = np.array([])
        self.fs = 1
        self.spectrograms = {}
        self.current_nperseg = 4096
        self.t_max = 1
        self.f_max = 1
        self.heatmap_items = []
        self.setup_ui()

    def setup_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('Datei')
        open_action = QAction('Öffnen...', self)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)

        ctrl_layout.addWidget(QLabel("Wasserfall dB Range:"))
        h_db = QHBoxLayout()
        self.slider_db_min = QSlider(Qt.Horizontal)
        self.slider_db_min.setRange(-150, 50)
        self.slider_db_min.setValue(-100)
        self.slider_db_min.valueChanged.connect(self.update_db_levels)

        self.slider_db_max = QSlider(Qt.Horizontal)
        self.slider_db_max.setRange(-150, 50)
        self.slider_db_max.setValue(0)
        self.slider_db_max.valueChanged.connect(self.update_db_levels)

        h_db.addWidget(QLabel("Min"))
        h_db.addWidget(self.slider_db_min)
        h_db.addWidget(QLabel("Max"))
        h_db.addWidget(self.slider_db_max)
        ctrl_layout.addLayout(h_db)

        ctrl_layout.addWidget(QLabel("Geschwindigkeit (Baud):"))
        h_speed = QHBoxLayout()
        self.spin_s_min = QSpinBox()
        self.spin_s_min.setRange(10, 100000)
        self.spin_s_min.setValue(50)
        self.spin_s_max = QSpinBox()
        self.spin_s_max.setRange(10, 100000)
        self.spin_s_max.setValue(5000)
        h_speed.addWidget(QLabel("Min:"))
        h_speed.addWidget(self.spin_s_min)
        h_speed.addWidget(QLabel("Max:"))
        h_speed.addWidget(self.spin_s_max)
        ctrl_layout.addLayout(h_speed)

        ctrl_layout.addWidget(QLabel("Anzahl paralleler Frequenzen:"))
        self.spin_freqs = QSpinBox()
        self.spin_freqs.setRange(1, 100)
        self.spin_freqs.setValue(5)
        ctrl_layout.addWidget(self.spin_freqs)

        self.chk_equidistant = QCheckBox("Äquidistante Frequenzen (FDM)")
        self.chk_equidistant.setChecked(True)
        ctrl_layout.addWidget(self.chk_equidistant)

        ctrl_layout.addWidget(QLabel("Zeige Top X % Ergebnisse:"))
        self.spin_top = QSpinBox()
        self.spin_top.setRange(1, 100)
        self.spin_top.setValue(10)
        ctrl_layout.addWidget(self.spin_top)

        ctrl_layout.addWidget(QLabel("Modulationen:"))
        self.list_mod = QListWidget()
        self.list_mod.setSelectionMode(QAbstractItemView.MultiSelection)
        self.list_mod.addItems(["2-PSK", "4-PSK", "8-PSK", "16-QAM", "64-QAM"])
        for i in range(self.list_mod.count()): self.list_mod.item(i).setSelected(True)
        ctrl_layout.addWidget(self.list_mod)

        self.btn_scan = QPushButton("Start SIGINT Scan")
        self.btn_scan.clicked.connect(self.start_scan)
        ctrl_layout.addWidget(self.btn_scan)

        self.lbl_status = QLabel("Bereit")
        ctrl_layout.addWidget(self.lbl_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        ctrl_layout.addWidget(self.progress_bar)

        self.lbl_nperseg = QLabel("FFT Size: -")
        ctrl_layout.addWidget(self.lbl_nperseg)

        ctrl_layout.addStretch()
        splitter.addWidget(ctrl_widget)

        pg.setConfigOptions(imageAxisOrder='row-major')
        self.plot_widget = pg.GraphicsLayoutWidget()

        vb_combined = CustomViewBox()
        self.plot_combined = self.plot_widget.addPlot(row=0, col=0, viewBox=vb_combined)
        self.plot_combined.setLabel('left', 'Frequenz', units='Hz')
        self.plot_combined.setLabel('bottom', 'Zeit', units='s')

        self.img_fft = pg.ImageItem()
        self.img_fft.setZValue(0)
        self.plot_combined.addItem(self.img_fft)

        colormap_fft = pg.colormap.get('inferno')
        self.img_fft.setLookupTable(colormap_fft.getLookupTable())

        self.plot_combined.getViewBox().sigRangeChanged.connect(self.update_fft_size)

        splitter.addWidget(self.plot_widget)
        splitter.setSizes([300, 1300])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            self.load_audio_file(filepath)

    def open_file_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Audiodatei öffnen", "", "All Files (*)")
        if filepath:
            self.load_audio_file(filepath)

    def load_audio_file(self, filepath):
        self.lbl_status.setText("Lese Audiodatei...")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        self.loader = AudioLoader(filepath)
        self.loader.finished.connect(self.process_audio)
        self.loader.error.connect(self.handle_load_error)
        self.loader.start()

    def handle_load_error(self):
        self.lbl_status.setText("Bereit")
        self.progress_bar.hide()

    def update_db_levels(self):
        db_min = self.slider_db_min.value()
        db_max = self.slider_db_max.value()
        if db_min >= db_max:
            self.slider_db_min.blockSignals(True)
            self.slider_db_min.setValue(db_max - 1)
            self.slider_db_min.blockSignals(False)

        if self.current_nperseg in self.spectrograms:
            self.apply_spectrogram()

    def process_audio(self, data, fs):
        self.audio_data = data
        self.fs = fs
        self.t_max = len(data) / fs
        self.f_max = fs / 2

        self.lbl_status.setText("Berechne STFT-Ebenen...")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.spec_worker = SpectrogramWorker(data, fs)
        self.spec_worker.progress.connect(self.progress_bar.setValue)
        self.spec_worker.finished.connect(self.store_spectrograms)
        self.spec_worker.start()

    def store_spectrograms(self, results):
        self.spectrograms = results
        self.lbl_status.setText("STFT vorberechnet. Bereit.")
        self.progress_bar.hide()
        self.current_nperseg = 1024

        sxx_ref = results[4096]['Sxx_log']
        abs_min = int(np.min(sxx_ref))
        abs_max = int(np.max(sxx_ref))
        noise_floor = int(np.percentile(sxx_ref, 5))

        self.slider_db_min.blockSignals(True)
        self.slider_db_max.blockSignals(True)

        self.slider_db_min.setRange(abs_min, abs_max)
        self.slider_db_max.setRange(abs_min, abs_max)

        self.slider_db_min.setValue(noise_floor)
        self.slider_db_max.setValue(abs_max)

        self.slider_db_min.blockSignals(False)
        self.slider_db_max.blockSignals(False)

        self.plot_combined.setLimits(xMin=0, xMax=self.t_max, yMin=0, yMax=self.f_max)

        self.apply_spectrogram()
        self.plot_combined.setRange(xRange=[0, self.t_max], yRange=[0, self.f_max])

    def update_fft_size(self, window, viewRange):
        if not self.spectrograms:
            return

        x_range, y_range = viewRange
        t_visible = max(1e-9, x_range[1] - x_range[0])
        f_visible = max(1e-9, y_range[1] - y_range[0])

        t_zoom = self.t_max / t_visible
        f_zoom = self.f_max / f_visible

        ratio = f_zoom / t_zoom
        shift = int(np.round(np.log2(ratio)))

        target_idx = np.clip(3 + shift, 0, 6)
        sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
        target_size = sizes[target_idx]

        if self.current_nperseg != target_size:
            self.current_nperseg = target_size
            self.apply_spectrogram()

    def apply_spectrogram(self):
        data = self.spectrograms[self.current_nperseg]
        img_data = data['Sxx_log']
        levels = (self.slider_db_min.value(), self.slider_db_max.value())
        self.img_fft.setImage(img_data, autoLevels=False, levels=levels)
        self.img_fft.setRect(pg.QtCore.QRectF(0, 0, self.t_max, self.f_max))
        self.lbl_nperseg.setText(f"FFT Size: {self.current_nperseg}")

    def start_scan(self):
        if not self.spectrograms: return
        selected_mods = [item.text() for item in self.list_mod.selectedItems()]

        base_data = self.spectrograms[4096]

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        equidistant_mode = self.chk_equidistant.isChecked()

        self.scanner = BruteForceScanner(
            self.audio_data, self.fs, base_data['Sxx_log'], base_data['f'],
            self.spin_s_min.value(), self.spin_s_max.value(),
            self.spin_freqs.value(), selected_mods, self.spin_top.value(),
            equidistant_mode
        )
        self.scanner.progress.connect(self.progress_bar.setValue)
        self.scanner.progress.connect(lambda p: self.lbl_status.setText(f"Berechnung: {p}%"))
        self.scanner.finished.connect(self.draw_detections)
        self.btn_scan.setEnabled(False)
        self.scanner.start()

    def draw_detections(self, detections):
        self.progress_bar.hide()
        for item in self.heatmap_items:
            self.plot_combined.removeItem(item)
        self.heatmap_items.clear()

        self.lbl_status.setText(f"Fertig. {len(detections)} Signale gefunden.")
        self.btn_scan.setEnabled(True)

        for det in detections:
            f_start = det['f_center'] - (det['bandwidth'] / 2)
            tooltip_text = f"Modulation: {det['mod']}\nBaudrate: {det['speed']:.1f} Bd\nGüte: {det['guete']:.1f}%\nCenter: {det['f_center']:.1f} Hz"

            rect = HoverRectItem(det['t_start'], f_start, det['duration'], det['bandwidth'], tooltip_text)
            self.plot_combined.addItem(rect)
            self.heatmap_items.append(rect)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())