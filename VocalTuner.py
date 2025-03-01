import os
import json
import wave
import datetime
import numpy as np
import pyaudio
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import math
import traceback
import queue
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Audio configuration
CHUNK = 2048             # Frames per buffer
FORMAT = pyaudio.paInt16 # 16-bit int format
CHANNELS = 1             # Mono

# Frequency range limits for pitch detection (typical human voice)
MIN_FREQ = 80.0   # Hz
MAX_FREQ = 400.0  # Hz

# Silence amplitude threshold
SILENCE_AMPLITUDE = 0.001

def freq_to_note_name(f):
    """Convert frequency (Hz) to the nearest musical note (A4=440Hz)."""
    try:
        if f is None or f <= 0 or math.isnan(f):
            return "0"
        n = round(12 * np.log2(f / 440.0))
        note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        note_index = n % 12
        octave = 4 + (n // 12)
        return note_names[note_index] + str(octave)
    except Exception:
        traceback.print_exc()
        return "0"

def detect_pitch_and_hnr(signal, rate, sensitivity):
    """
    Uses autocorrelation to estimate pitch and HNR.
    If the RMS amplitude is below the sensitivity threshold, returns (None, np.nan).
    """
    try:
        rms = np.sqrt(np.mean(signal**2))
        if rms < sensitivity:
            return (None, np.nan)
        x = signal - np.mean(signal)
        corr = np.correlate(x, x, mode='full')[len(x)-1:]
        min_lag = int(rate / MAX_FREQ)
        max_lag = int(rate / MIN_FREQ)
        if max_lag > len(corr):
            max_lag = len(corr)
        if max_lag <= min_lag:
            return (None, np.nan)
        segment = corr[min_lag:max_lag]
        peak_index = np.argmax(segment) + min_lag
        if peak_index == 0:
            return (None, np.nan)
        pitch = rate / peak_index
        noise = (np.sum(segment) - corr[peak_index]) / (len(segment) - 1)
        hnr = 10 * np.log10(corr[peak_index] / noise) if noise > 0 else np.nan
        return (pitch, hnr)
    except Exception:
        traceback.print_exc()
        return (None, np.nan)

def levinson_durbin(r, order):
    """Robust Levinson-Durbin algorithm for LPC estimation."""
    try:
        a = np.zeros(order+1)
        e = r[0]
        a[0] = 1.0
        for i in range(1, order+1):
            acc = 0.0
            for j in range(1, i):
                acc += a[j] * r[i-j]
            k = (r[i] - acc) / e if abs(e) > 1e-6 else 0.0
            a[i] = k
            for j in range(1, i):
                a[j] -= k * a[i-j]
            e *= (1 - k * k)
            if e < 1e-6:
                break
        return a
    except Exception:
        traceback.print_exc()
        return np.zeros(order+1)

def compute_formants(signal, rate, lpc_order=16):
    """
    Compute up to three formant frequencies using LPC.
    Applies pre-emphasis and a Hamming window.
    Returns a list of three values (F1, F2, F3) in Hz; if fewer are found, pads with np.nan.
    Only frequencies between 0 and 4000 Hz are considered.
    """
    try:
        pre_emphasis = 0.97
        emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        windowed = emphasized * np.hamming(len(emphasized))
        corr = np.correlate(windowed, windowed, mode='full')[len(windowed)-1:]
        if len(corr) < lpc_order + 1:
            return [np.nan, np.nan, np.nan]
        r = corr[:lpc_order+1]
        if abs(r[0]) < 1e-6:
            return [np.nan, np.nan, np.nan]
        A = levinson_durbin(r, lpc_order)
        roots = np.roots(A)
        roots = [root for root in roots if np.imag(root) > 0]
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * (rate / (2 * np.pi))
        formants = sorted([f for f in freqs if 0 < f < 4000])
        while len(formants) < 3:
            formants.append(np.nan)
        return formants[:3]
    except Exception:
        traceback.print_exc()
        return [np.nan, np.nan, np.nan]

###############################################################################
# Audio capture using PyAudio callback mode with a result queue
###############################################################################

class AudioStream:
    def __init__(self, device_index, rate, record_audio, sensitivity, result_queue):
        self.device_index = device_index
        self.rate = rate
        self.record_audio = record_audio
        self.sensitivity = sensitivity
        self.result_queue = result_queue  # Queue for sending results to main thread
        self.recorded_frames = []
        self.pa = pyaudio.PyAudio()
        try:
            self.stream = self.pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK,
                stream_callback=self.callback
            )
            self.stream.start_stream()
            self.sample_width = self.pa.get_sample_size(FORMAT)
            print(f"Audio stream opened on device {self.device_index} at {self.rate} Hz (callback mode).")
        except Exception as e:
            print("Error opening stream:", e)
            traceback.print_exc()
            self.stream = None

    def callback(self, in_data, frame_count, time_info, status_flags):
        try:
            if self.record_audio:
                self.recorded_frames.append(in_data)
            if not in_data:
                return (in_data, pyaudio.paContinue)
            audio_data = np.frombuffer(in_data, np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(audio_data**2))
            amplitude = rms / 32768.0
            pitch_val, hnr_val = detect_pitch_and_hnr(audio_data, self.rate, self.sensitivity)
            # If amplitude is below threshold or no valid pitch, force to 0.
            if amplitude < SILENCE_AMPLITUDE or pitch_val is None:
                pitch_val = 0
                note = "0"
            else:
                note = freq_to_note_name(pitch_val).upper()
            formants_val = compute_formants(audio_data, self.rate)
            self.result_queue.put((note, pitch_val, formants_val, hnr_val, amplitude))
        except Exception as e:
            print("Error in callback:")
            traceback.print_exc()
            self.result_queue.put(("N/A", 0, [np.nan, np.nan, np.nan], np.nan, 0.0))
        return (in_data, pyaudio.paContinue)

    def stop(self):
        try:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
            self.pa.terminate()
            print("Audio stream stopped.")
        except Exception as e:
            print("Error stopping audio stream:")
            traceback.print_exc()

###############################################################################
# UI and Application
###############################################################################

class App:
    def __init__(self, root):
        self.root = root
        root.title("Transgender Voice Training Tool")
        # Setup default output directory in %APPDATA%\VocalTuner
        self.default_output = os.path.join(os.getenv("APPDATA"), "VocalTuner")
        os.makedirs(self.default_output, exist_ok=True)
        self.output_dir = self.default_output
        self.config_path = os.path.join(self.output_dir, "config.json")
        self.metrics_log_path = os.path.join(self.output_dir, "metrics.log")
        # Load configuration
        self.selected_mic = tk.StringVar()
        self.recording = tk.BooleanVar(value=False)
        self.sensitivity = 0.01  # default mic sensitivity (RMS threshold)
        self.reset_interval_value = 3.0  # default reset interval (seconds)
        self.load_config()
        # Create a thread-safe queue for audio results
        self.result_queue = queue.Queue()
        # Store last valid tone and pitch
        self.last_tone = "0"
        self.last_pitch = 0
        # Initialize formant smoothing variables (one for each formant)
        self.formant_smoothing_factor = 0.1
        self.last_formants = [0, 0, 0]
        # Menus
        self.menubar = tk.Menu(root)
        root.config(menu=self.menubar)
        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label="Mic Settings", command=self.open_mic_settings)
        file_menu.add_command(label="Set Output Directory", command=self.set_output_directory)
        file_menu.add_checkbutton(label="Record Audio", variable=self.recording)
        self.menubar.add_cascade(label="File", menu=file_menu)
        settings_menu = tk.Menu(self.menubar, tearoff=0)
        settings_menu.add_command(label="Set Mic Sensitivity", command=self.set_mic_sensitivity)
        settings_menu.add_command(label="Set Reset Interval", command=self.set_reset_interval)
        self.menubar.add_cascade(label="Settings", menu=settings_menu)
        ranges_menu = tk.Menu(self.menubar, tearoff=0)
        ranges_menu.add_command(label="View Transgender Pitch Ranges", command=self.show_ranges)
        self.menubar.add_cascade(label="Ranges", menu=ranges_menu)
        # Start/Stop Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(padx=10, pady=5)
        self.start_button = tk.Button(btn_frame, text="Start", command=self.start_listening)
        self.start_button.pack(side=tk.LEFT, padx=10)
        self.stop_button = tk.Button(btn_frame, text="Stop", command=self.stop_listening, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        # Tone display area (smaller)
        self.tone_frame = tk.Frame(root, width=300, height=50, bd=2, relief=tk.SUNKEN)
        self.tone_frame.pack(padx=10, pady=5)
        self.tone_frame.pack_propagate(False)
        self.tone_label = tk.Label(self.tone_frame, text="0", font=("Helvetica", 24))
        self.tone_label.pack(expand=True)
        self.quality_label = tk.Label(root, text="Jitter: N/A | Shimmer: N/A | HNR: N/A", font=("Helvetica", 12))
        self.quality_label.pack(padx=10, pady=5)
        self.metrics_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
        self.metrics_frame.pack(padx=10, pady=5, fill=tk.X)
        self.pitch_summary_label = tk.Label(self.metrics_frame, text="Pitch Summary: Low: N/A | High: N/A | Avg: N/A", font=("Helvetica", 12))
        self.pitch_summary_label.pack(padx=10, pady=2)
        self.formant_summary_label = tk.Label(self.metrics_frame, text="Formant Summary: F1: N/A | F2: N/A | F3: N/A", font=("Helvetica", 12))
        self.formant_summary_label.pack(padx=10, pady=2)
        self.pitch_history = []
        self.amplitude_history = []
        self.formant_histories = [[], [], []]  # For F1, F2, F3
        self.metrics_pitch_history = []
        self.metrics_formant_histories = [[], [], []]
        self.fig, (self.ax_pitch, self.ax_formants) = plt.subplots(2, 1, figsize=(6,4), sharex=True)
        self.ax_pitch.set_title("Live Pitch History")
        self.ax_pitch.set_ylabel("Frequency (Hz)")
        self.pitch_line, = self.ax_pitch.plot([], [], color='blue', label="Pitch")
        self.ax_pitch.legend(loc='upper right')
        self.ax_pitch.grid(True)
        self.ax_formants.set_title("Live Formant History (F1, F2, F3)")
        self.ax_formants.set_xlabel("Time (frames)")
        self.ax_formants.set_ylabel("Frequency (Hz)")
        colors = ['red', 'green', 'orange']
        self.formant_lines = []
        for i in range(3):
            line, = self.ax_formants.plot([], [], color=colors[i], label=f"F{i+1}")
            self.formant_lines.append(line)
        self.ax_formants.legend(loc='upper right')
        self.ax_formants.grid(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(padx=10, pady=5)
        self.audio_stream = None
        self.reset_timer = None
        self.schedule_reset()
        self.poll_queue()

    def load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                self.selected_mic.set(config.get("mic", ""))
                self.recording.set(config.get("record_audio", False))
                self.sensitivity = config.get("sensitivity", 0.01)
                self.reset_interval_value = config.get("reset_interval", 3.0)
                out_dir = config.get("output_dir", "")
                if out_dir:
                    self.output_dir = out_dir
                    os.makedirs(self.output_dir, exist_ok=True)
                    self.metrics_log_path = os.path.join(self.output_dir, "metrics.log")
        except Exception:
            traceback.print_exc()

    def save_config(self):
        try:
            config = {
                "mic": self.selected_mic.get(),
                "record_audio": self.recording.get(),
                "sensitivity": self.sensitivity,
                "reset_interval": self.reset_interval_value,
                "output_dir": self.output_dir
            }
            with open(self.config_path, "w") as f:
                json.dump(config, f)
            print("Config saved.")
        except Exception:
            traceback.print_exc()

    def get_mic_devices(self):
        try:
            pa = pyaudio.PyAudio()
            devices = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    devices.append((i, info.get("name", "Unknown")))
            pa.terminate()
            return devices
        except Exception:
            traceback.print_exc()
            return []

    def open_mic_settings(self):
        mic_win = tk.Toplevel(self.root)
        mic_win.title("Mic Settings")
        tk.Label(mic_win, text="Select Microphone:").pack(padx=10, pady=5)
        devices = self.get_mic_devices()
        mic_names = [name for idx, name in devices]
        sel = tk.StringVar()
        sel.set(self.selected_mic.get() if self.selected_mic.get() in mic_names else (mic_names[0] if mic_names else ""))
        dropdown = tk.OptionMenu(mic_win, sel, *mic_names)
        dropdown.pack(padx=10, pady=5)
        def save_mic():
            self.selected_mic.set(sel.get())
            self.save_config()
            mic_win.destroy()
        tk.Button(mic_win, text="Save", command=save_mic).pack(padx=10, pady=10)

    def set_output_directory(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = directory
            os.makedirs(self.output_dir, exist_ok=True)
            self.metrics_log_path = os.path.join(self.output_dir, "metrics.log")
            self.save_config()
            messagebox.showinfo("Output Directory", f"Files will be saved to:\n{self.output_dir}")

    def set_mic_sensitivity(self):
        new_val = simpledialog.askfloat("Mic Sensitivity", "Enter mic sensitivity (RMS threshold):", initialvalue=self.sensitivity)
        if new_val is not None:
            self.sensitivity = new_val
            self.save_config()

    def set_reset_interval(self):
        new_val = simpledialog.askfloat("Reset Interval", "Enter reset interval (sec):", initialvalue=self.reset_interval_value)
        if new_val is not None:
            self.reset_interval_value = new_val
            self.save_config()

    def show_ranges(self):
        ranges_text = (
            "Transgender Pitch Ranges:\n"
            "• Gender neutral: 150–185 Hz\n"
            "• Vocal feminization: Aim ~180 Hz\n"
            "• Vocal masculinization: Aim ~150 Hz\n"
            "• Average female pitch: ~225 Hz\n"
            "• Average male pitch: ~125 Hz\n"
            "Consult a speech therapist for personalized guidance."
        )
        messagebox.showinfo("Transgender Pitch Ranges", ranges_text)

    def update_ui(self, note, pitch, formants, hnr, amplitude):
        try:
            # If amplitude is below threshold, treat as silence.
            if amplitude < SILENCE_AMPLITUDE:
                display_tone = "0"
                pitch_to_append = 0
            else:
                if pitch is None:
                    display_tone = self.last_tone if hasattr(self, "last_tone") else "0"
                    pitch_to_append = self.last_pitch if hasattr(self, "last_pitch") else 0
                else:
                    display_tone = note.upper()
                    self.last_tone = display_tone
                    self.last_pitch = pitch
                    pitch_to_append = pitch

            self.tone_label.config(text=display_tone)
            self.pitch_history.append(pitch_to_append)
            if len(self.pitch_history) > 100:
                self.pitch_history = self.pitch_history[-100:]
            self.pitch_line.set_data(range(len(self.pitch_history)), self.pitch_history)
            self.ax_pitch.relim()
            self.ax_pitch.autoscale_view()

            self.amplitude_history.append(amplitude)
            if len(self.amplitude_history) > 100:
                self.amplitude_history = self.amplitude_history[-100:]
            # Smooth formant values with exponential smoothing.
            if not hasattr(self, "last_formants"):
                self.last_formants = [0, 0, 0]
            smoothed_formants = []
            for i in range(3):
                new_val = formants[i] if len(formants) > i else np.nan
                # Only update if new value is finite; otherwise, keep old.
                if np.isfinite(new_val):
                    # If last value is not finite, initialize it.
                    if not np.isfinite(self.last_formants[i]):
                        smoothed = new_val
                    else:
                        smoothed = self.formant_smoothing_factor * new_val + (1 - self.formant_smoothing_factor) * self.last_formants[i]
                else:
                    smoothed = self.last_formants[i] if np.isfinite(self.last_formants[i]) else 0
                self.last_formants[i] = smoothed
                smoothed_formants.append(smoothed)
                self.formant_histories[i].append(smoothed)
                if len(self.formant_histories[i]) > 100:
                    self.formant_histories[i] = self.formant_histories[i][-100:]
                self.formant_lines[i].set_data(range(len(self.formant_histories[i])), self.formant_histories[i])
            self.ax_formants.relim()
            self.ax_formants.autoscale_view()
            self.canvas.draw_idle()

            periods = [1.0 / p for p in self.pitch_history if p and p > 0 and not np.isnan(p)]
            jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100 if len(periods) >= 2 else np.nan
            valid_amps = [a for a in self.amplitude_history if a and not np.isnan(a)]
            shimmer = np.mean(np.abs(np.diff(valid_amps))) / np.mean(valid_amps) * 100 if len(valid_amps) >= 2 else np.nan
            self.quality_label.config(text=f"Jitter: {jitter:.2f}% | Shimmer: {shimmer:.2f}% | HNR: {hnr:.2f} dB")

            if pitch is not None and pitch > 0:
                self.metrics_pitch_history.append(pitch)
            for i in range(3):
                if len(formants) > i and formants[i] > 0:
                    self.metrics_formant_histories[i].append(formants[i])
                else:
                    self.metrics_formant_histories[i].append(np.nan)
        except Exception:
            traceback.print_exc()

    def poll_queue(self):
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self.update_ui(*result)
        except Exception:
            traceback.print_exc()
        self.root.after(50, self.poll_queue)

    def start_listening(self):
        try:
            devices = self.get_mic_devices()
            device_index = None
            for idx, name in devices:
                if name == self.selected_mic.get():
                    device_index = idx
                    break
            if device_index is None:
                if devices:
                    device_index = devices[0][0]
                    self.selected_mic.set(devices[0][1])
                else:
                    print("No microphone found!")
                    return
            pa = pyaudio.PyAudio()
            try:
                info = pa.get_device_info_by_index(device_index)
                device_rate = int(info.get("defaultSampleRate", 44100))
            except Exception:
                device_rate = 44100
            pa.terminate()
            print(f"Starting audio on '{self.selected_mic.get()}' (device {device_index}) at {device_rate} Hz.")
            self.audio_stream = AudioStream(device_index, device_rate, self.recording.get(), self.sensitivity, self.result_queue)
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        except Exception:
            traceback.print_exc()

    def stop_listening(self):
        try:
            if self.audio_stream:
                self.audio_stream.stop()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.tone_label.config(text="N/A")
            if self.recording.get() and self.audio_stream and self.audio_stream.recorded_frames:
                self.save_recording()
            self.save_full_chart()
        except Exception:
            traceback.print_exc()

    def save_recording(self):
        try:
            if not (self.audio_stream and self.audio_stream.recorded_frames):
                print("No audio recorded.")
                return
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"recording_{timestamp}.wav")
            wf = wave.open(filename, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio_stream.sample_width)
            wf.setframerate(self.audio_stream.rate)
            wf.writeframes(b"".join(self.audio_stream.recorded_frames))
            wf.close()
            print(f"Audio recording saved to {filename}")
        except Exception:
            traceback.print_exc()

    def save_full_chart(self):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"full_chart_{timestamp}.png")
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            ax1.set_title("Full Session Pitch History")
            ax1.set_ylabel("Frequency (Hz)")
            ax1.plot(range(len(self.pitch_history)), self.pitch_history, color='blue', label="Pitch")
            ax1.legend(loc='upper right')
            ax1.grid(True)
            ax2.set_title("Full Session Formant History (F1, F2, F3)")
            ax2.set_xlabel("Time (frames)")
            ax2.set_ylabel("Frequency (Hz)")
            colors = ['red', 'green', 'orange']
            for i in range(3):
                ax2.plot(range(len(self.formant_histories[i])), self.formant_histories[i], color=colors[i], label=f"F{i+1}")
            ax2.legend(loc='upper right')
            ax2.grid(True)
            fig2.tight_layout()
            fig2.savefig(filename)
            plt.close(fig2)
            print(f"Full chart image saved to {filename}")
        except Exception:
            traceback.print_exc()

    def reset_interval_metrics(self):
        try:
            if self.metrics_pitch_history:
                arr = np.array(self.metrics_pitch_history)
                valid = arr[arr > 0]
                if valid.size > 0:
                    low = np.nanmin(valid)
                    high = np.nanmax(valid)
                    avg = np.nanmean(valid)
                    pitch_text = f"Low: {low:.1f} Hz | High: {high:.1f} Hz | Avg: {avg:.1f} Hz"
                else:
                    pitch_text = "Low: N/A | High: N/A | Avg: N/A"
            else:
                pitch_text = "Low: N/A | High: N/A | Avg: N/A"
            formant_text = ""
            for i in range(3):
                arr = np.array(self.metrics_formant_histories[i])
                valid = arr[arr > 0]
                if valid.size == 0:
                    formant_text += f"F{i+1}: Low: N/A, High: N/A, Avg: N/A   "
                else:
                    low = np.nanmin(valid)
                    high = np.nanmax(valid)
                    avg = np.nanmean(valid)
                    formant_text += f"F{i+1}: Low: {low:.1f} Hz, High: {high:.1f} Hz, Avg: {avg:.1f} Hz   "
            self.pitch_summary_label.config(text="Pitch Summary: " + pitch_text)
            self.formant_summary_label.config(text="Formant Summary: " + formant_text)
            log_line = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Pitch: {pitch_text} | Formants: {formant_text}\n"
            with open(self.metrics_log_path, "a") as f:
                f.write(log_line)
        except Exception:
            traceback.print_exc()

    def schedule_reset(self):
        try:
            self.reset_interval_metrics()
            self.reset_timer = self.root.after(int(self.reset_interval_value * 1000), self.schedule_reset)
        except Exception:
            traceback.print_exc()

    def on_close(self):
        try:
            if self.reset_timer is not None:
                self.root.after_cancel(self.reset_timer)
            if self.audio_stream:
                self.audio_stream.stop()
            self.root.destroy()
        except Exception:
            traceback.print_exc()

    def poll_queue(self):
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self.update_ui(*result)
        except Exception:
            traceback.print_exc()
        self.root.after(50, self.poll_queue)

    def start_listening(self):
        try:
            devices = self.get_mic_devices()
            device_index = None
            for idx, name in devices:
                if name == self.selected_mic.get():
                    device_index = idx
                    break
            if device_index is None:
                if devices:
                    device_index = devices[0][0]
                    self.selected_mic.set(devices[0][1])
                else:
                    print("No microphone found!")
                    return
            pa = pyaudio.PyAudio()
            try:
                info = pa.get_device_info_by_index(device_index)
                device_rate = int(info.get("defaultSampleRate", 44100))
            except Exception:
                device_rate = 44100
            pa.terminate()
            print(f"Starting audio on '{self.selected_mic.get()}' (device {device_index}) at {device_rate} Hz.")
            self.audio_stream = AudioStream(device_index, device_rate, self.recording.get(), self.sensitivity, self.result_queue)
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        except Exception:
            traceback.print_exc()

    def stop_listening(self):
        try:
            if self.audio_stream:
                self.audio_stream.stop()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.tone_label.config(text="N/A")
            if self.recording.get() and self.audio_stream and self.audio_stream.recorded_frames:
                self.save_recording()
            self.save_full_chart()
        except Exception:
            traceback.print_exc()

    def save_recording(self):
        try:
            if not (self.audio_stream and self.audio_stream.recorded_frames):
                print("No audio recorded.")
                return
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"recording_{timestamp}.wav")
            wf = wave.open(filename, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio_stream.sample_width)
            wf.setframerate(self.audio_stream.rate)
            wf.writeframes(b"".join(self.audio_stream.recorded_frames))
            wf.close()
            print(f"Audio recording saved to {filename}")
        except Exception:
            traceback.print_exc()

    def save_full_chart(self):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"full_chart_{timestamp}.png")
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            ax1.set_title("Full Session Pitch History")
            ax1.set_ylabel("Frequency (Hz)")
            ax1.plot(range(len(self.pitch_history)), self.pitch_history, color='blue', label="Pitch")
            ax1.legend(loc='upper right')
            ax1.grid(True)
            ax2.set_title("Full Session Formant History (F1, F2, F3)")
            ax2.set_xlabel("Time (frames)")
            ax2.set_ylabel("Frequency (Hz)")
            colors = ['red', 'green', 'orange']
            for i in range(3):
                ax2.plot(range(len(self.formant_histories[i])), self.formant_histories[i], color=colors[i], label=f"F{i+1}")
            ax2.legend(loc='upper right')
            ax2.grid(True)
            fig2.tight_layout()
            fig2.savefig(filename)
            plt.close(fig2)
            print(f"Full chart image saved to {filename}")
        except Exception:
            traceback.print_exc()

    def reset_interval_metrics(self):
        try:
            if self.metrics_pitch_history:
                arr = np.array(self.metrics_pitch_history)
                valid = arr[arr > 0]
                if valid.size > 0:
                    low = np.nanmin(valid)
                    high = np.nanmax(valid)
                    avg = np.nanmean(valid)
                    pitch_text = f"Low: {low:.1f} Hz | High: {high:.1f} Hz | Avg: {avg:.1f} Hz"
                else:
                    pitch_text = "Low: N/A | High: N/A | Avg: N/A"
            else:
                pitch_text = "Low: N/A | High: N/A | Avg: N/A"
            formant_text = ""
            for i in range(3):
                arr = np.array(self.metrics_formant_histories[i])
                valid = arr[arr > 0]
                if valid.size == 0:
                    formant_text += f"F{i+1}: Low: N/A, High: N/A, Avg: N/A   "
                else:
                    low = np.nanmin(valid)
                    high = np.nanmax(valid)
                    avg = np.nanmean(valid)
                    formant_text += f"F{i+1}: Low: {low:.1f} Hz, High: {high:.1f} Hz, Avg: {avg:.1f} Hz   "
            self.pitch_summary_label.config(text="Pitch Summary: " + pitch_text)
            self.formant_summary_label.config(text="Formant Summary: " + formant_text)
            log_line = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Pitch: {pitch_text} | Formants: {formant_text}\n"
            with open(self.metrics_log_path, "a") as f:
                f.write(log_line)
        except Exception:
            traceback.print_exc()

    def schedule_reset(self):
        try:
            self.reset_interval_metrics()
            self.reset_timer = self.root.after(int(self.reset_interval_value * 1000), self.schedule_reset)
        except Exception:
            traceback.print_exc()

    def on_close(self):
        try:
            if self.reset_timer is not None:
                self.root.after_cancel(self.reset_timer)
            if self.audio_stream:
                self.audio_stream.stop()
            self.root.destroy()
        except Exception:
            traceback.print_exc()

    def poll_queue(self):
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self.update_ui(*result)
        except Exception:
            traceback.print_exc()
        self.root.after(50, self.poll_queue)

    def start_listening(self):
        try:
            devices = self.get_mic_devices()
            device_index = None
            for idx, name in devices:
                if name == self.selected_mic.get():
                    device_index = idx
                    break
            if device_index is None:
                if devices:
                    device_index = devices[0][0]
                    self.selected_mic.set(devices[0][1])
                else:
                    print("No microphone found!")
                    return
            pa = pyaudio.PyAudio()
            try:
                info = pa.get_device_info_by_index(device_index)
                device_rate = int(info.get("defaultSampleRate", 44100))
            except Exception:
                device_rate = 44100
            pa.terminate()
            print(f"Starting audio on '{self.selected_mic.get()}' (device {device_index}) at {device_rate} Hz.")
            self.audio_stream = AudioStream(device_index, device_rate, self.recording.get(), self.sensitivity, self.result_queue)
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        except Exception:
            traceback.print_exc()

    def stop_listening(self):
        try:
            if self.audio_stream:
                self.audio_stream.stop()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.tone_label.config(text="N/A")
            if self.recording.get() and self.audio_stream and self.audio_stream.recorded_frames:
                self.save_recording()
            self.save_full_chart()
        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
