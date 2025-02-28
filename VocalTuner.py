import numpy as np
import pyaudio
import tkinter as tk
import threading
import time
import math

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Audio configuration
CHUNK = 2048             # Audio samples per frame
FORMAT = pyaudio.paInt16 # 16-bit resolution
CHANNELS = 1             # Mono

# Frequency range for pitch detection (in Hz)
MIN_FREQ = 80.0
MAX_FREQ = 1000.0

def freq_to_note_name(f):
    """Convert frequency (Hz) to the closest musical note using A4=440Hz."""
    if f is None or f <= 0 or math.isnan(f):
        return "N/A"
    n = round(12 * np.log2(f / 440.0))
    note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    note_index = n % 12
    octave = 4 + (n // 12)
    return note_names[note_index] + str(octave)

def detect_pitch_and_hnr(data, rate):
    """
    Estimate pitch using autocorrelation and compute a simple HNR.
    Returns (pitch in Hz, HNR in dB).
    """
    data = data - np.mean(data)
    corr = np.correlate(data, data, mode='full')
    corr = corr[len(corr)//2:]
    
    lag_min = int(rate / MAX_FREQ)
    lag_max = int(rate / MIN_FREQ)
    if lag_max >= len(corr):
        lag_max = len(corr) - 1
    if lag_min >= lag_max:
        return (None, np.nan)
    
    segment = corr[lag_min:lag_max]
    peak_index = np.argmax(segment) + lag_min
    if corr[peak_index] < 1e-6:
        return (None, np.nan)
    
    pitch = rate / peak_index
    
    if len(segment) <= 1:
        hnr = np.nan
    else:
        noise_value = (np.sum(segment) - corr[peak_index]) / (len(segment) - 1)
        hnr = 10 * np.log10(corr[peak_index] / noise_value) if noise_value > 0 else np.nan
    return (pitch, hnr)

def levinson_durbin(r, order):
    """Levinson-Durbin algorithm for LPC coefficient estimation."""
    a = np.zeros(order+1)
    e = r[0]
    a[0] = 1.0
    for i in range(1, order+1):
        acc = 0.0
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = (r[i] - acc) / e
        a[i] = k
        for j in range(1, i):
            a[j] = a[j] - k * a[i - j]
        e *= (1 - k * k)
    return a

def compute_formants(signal, rate, lpc_order=12):
    """
    Estimate formant frequencies using a basic LPC approach.
    Returns a list of formant frequencies.
    """
    pre_emphasis = 0.97
    emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    windowed = emphasized * np.hamming(len(emphasized))
    corr = np.correlate(windowed, windowed, mode='full')
    corr = corr[len(corr)//2:]
    if len(corr) < lpc_order+1:
        return []
    r = corr[:lpc_order+1]
    try:
        A = levinson_durbin(r, lpc_order)
    except Exception:
        return []
    roots = np.roots(A)
    # Only take one root from each conjugate pair (with positive imaginary part)
    roots = [r for r in roots if np.imag(r) >= 0.01]
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (rate / (2 * np.pi))
    # Only consider frequencies in a typical speech range (for formants)
    freqs = [f for f in freqs if 300 < f < 3000]
    freqs.sort()
    return freqs

class AudioThread(threading.Thread):
    """
    Reads audio, analyzes pitch, formants, and voice quality,
    then calls the update callback with:
    (note, pitch, formants, HNR, amplitude)
    """
    def __init__(self, update_callback, device_index, rate, get_sensitivity):
        super().__init__()
        self.update_callback = update_callback
        self.device_index = device_index
        self.rate = rate
        self.get_sensitivity = get_sensitivity
        self.running = True
        self.pa = pyaudio.PyAudio()
        try:
            self.stream = self.pa.open(format=FORMAT,
                                       channels=CHANNELS,
                                       rate=self.rate,
                                       input=True,
                                       input_device_index=self.device_index,
                                       frames_per_buffer=CHUNK)
            print(f"Audio stream opened on device {self.device_index} at {self.rate} Hz.")
        except Exception as e:
            print("Error opening stream:", e)
            self.running = False
        
    def run(self):
        while self.running:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, np.int16).astype(np.float32)
                rms_raw = np.sqrt(np.mean(audio_data**2))
                rms_normalized = rms_raw / 32768.0
                threshold = self.get_sensitivity()
                if rms_normalized < threshold:
                    note = "N/A"
                    pitch_val = None
                    hnr_val = np.nan
                    formants_val = []
                else:
                    norm_audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) != 0 else audio_data
                    pitch_val, hnr_val = detect_pitch_and_hnr(norm_audio_data, self.rate)
                    note = freq_to_note_name(pitch_val)
                    formants_val = compute_formants(norm_audio_data, self.rate)
                self.update_callback(note, pitch_val, formants_val, hnr_val, rms_normalized)
            except Exception as e:
                print("Error in audio thread:", e)
            time.sleep(0.01)
            
    def stop(self):
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
        print("Audio stream closed.")

class App:
    def __init__(self, root):
        self.root = root
        root.title("Transgender Voice Training Tool")

        # Guidance Label: Pitch Ranges for Transgender Voice Training
        guidance_text = (
            "Transgender Pitch Ranges:\n"
            "• Gender neutral: 150–185 Hz\n"
            "• Vocal feminization: Aim ~180 Hz\n"
            "• Vocal masculinization: Aim ~150 Hz\n"
            "• Average female pitch: ~225 Hz\n"
            "• Average male pitch: ~125 Hz\n"
            "Consult a speech therapist for personalized guidance."
        )
        self.guidance_label = tk.Label(root, text=guidance_text, font=("Helvetica", 10), justify="left")
        self.guidance_label.pack(padx=10, pady=5)

        # Retrieve available microphones
        self.mic_devices = self.get_mic_devices()
        self.mic_mapping = {}
        mic_names = []
        for idx, name in self.mic_devices:
            mic_names.append(name)
            self.mic_mapping[name] = idx

        # Mic selection dropdown
        self.selected_mic = tk.StringVar()
        self.selected_mic.set(mic_names[0] if mic_names else "No microphone found")
        self.dropdown = tk.OptionMenu(root, self.selected_mic, *mic_names)
        self.dropdown.config(width=30)
        self.dropdown.pack(padx=10, pady=5)

        # Sensitivity slider (RMS threshold) starting at 0.003 with manual entry
        self.sensitivity_value = tk.DoubleVar(value=0.003)
        sensitivity_frame = tk.Frame(root)
        sensitivity_frame.pack(padx=10, pady=5)
        tk.Label(sensitivity_frame, text="Input Sensitivity (RMS Threshold):").pack(side=tk.LEFT)
        self.sensitivity_slider = tk.Scale(sensitivity_frame, from_=0.0, to=0.2, resolution=0.001,
                                           orient=tk.HORIZONTAL, variable=self.sensitivity_value)
        self.sensitivity_slider.pack(side=tk.LEFT)
        # Entry widget to allow manual editing of the RMS threshold
        self.sensitivity_entry = tk.Entry(sensitivity_frame, textvariable=self.sensitivity_value, width=5)
        self.sensitivity_entry.pack(side=tk.LEFT, padx=5)

        # Start and Stop buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(padx=10, pady=5)
        self.start_button = tk.Button(btn_frame, text="Start", command=self.start_listening)
        self.start_button.pack(side=tk.LEFT, padx=10)
        self.stop_button = tk.Button(btn_frame, text="Stop", command=self.stop_listening, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        # Reset interval field (in seconds) for metrics summary
        reset_frame = tk.Frame(root)
        reset_frame.pack(padx=10, pady=5)
        tk.Label(reset_frame, text="Reset Interval (sec):").pack(side=tk.LEFT)
        self.reset_interval = tk.Entry(reset_frame, width=5)
        self.reset_interval.insert(0, "3")
        self.reset_interval.pack(side=tk.LEFT)

        # Live chart: tone display box
        self.tone_frame = tk.Frame(root, width=300, height=100, bd=2, relief=tk.SUNKEN)
        self.tone_frame.pack(padx=10, pady=5)
        self.tone_frame.pack_propagate(False)
        self.tone_label = tk.Label(self.tone_frame, text="N/A", font=("Helvetica", 48))
        self.tone_label.pack(expand=True)

        # Quality metrics label (jitter, shimmer, HNR) updating continuously
        self.quality_label = tk.Label(root, text="Jitter: N/A | Shimmer: N/A | HNR: N/A", font=("Helvetica", 12))
        self.quality_label.pack(padx=10, pady=5)

        # Metrics Summary Box (separate from the live chart)
        self.metrics_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
        self.metrics_frame.pack(padx=10, pady=5, fill=tk.X)
        self.pitch_summary_label = tk.Label(self.metrics_frame, text="Pitch Summary: Low: N/A | High: N/A | Avg: N/A", font=("Helvetica", 12))
        self.pitch_summary_label.pack(padx=10, pady=2)
        self.formant_summary_label = tk.Label(self.metrics_frame, text="Formant Summary: F1: N/A | F2: N/A", font=("Helvetica", 12))
        self.formant_summary_label.pack(padx=10, pady=2)

        # Initialize live history data for charts
        self.pitch_history = []
        self.amplitude_history = []
        self.formant_histories = [[], []]  # For F1 and F2

        # Initialize metrics data (for summary) that resets every interval
        self.metrics_pitch_history = []
        self.metrics_formant_histories = [[], []]

        # Combined chart with two subplots: live pitch on top, live formants (F1 & F2) on bottom
        self.fig, (self.ax_pitch, self.ax_formants) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
        self.ax_pitch.set_title("Live Pitch History")
        self.ax_pitch.set_ylabel("Frequency (Hz)")
        self.pitch_line, = self.ax_pitch.plot([], [], color='blue', label="Pitch")
        self.ax_pitch.legend(loc='upper right')
        self.ax_pitch.grid(True)

        self.ax_formants.set_title("Live Formant History (F1 & F2)")
        self.ax_formants.set_xlabel("Time")
        self.ax_formants.set_ylabel("Frequency (Hz)")
        colors = ['red', 'green']
        self.formant_lines = []
        for i in range(2):
            line, = self.ax_formants.plot([], [], color=colors[i], label=f"F{i+1}")
            self.formant_lines.append(line)
        self.ax_formants.legend(loc='upper right')
        self.ax_formants.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(padx=10, pady=5)

        self.audio_thread = None
        self.reset_timer = None
        self.schedule_reset()

    def get_mic_devices(self):
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                devices.append((i, info.get('name', 'Unknown')))
        pa.terminate()
        return devices

    def update_ui(self, note, pitch, formants, hnr, amplitude):
        # Update tone display (live)
        self.tone_label.config(text=note)

        # Update live chart histories (moving window)
        if pitch is None:
            self.pitch_history.append(np.nan)
        else:
            self.pitch_history.append(pitch)
        if len(self.pitch_history) > 100:
            self.pitch_history = self.pitch_history[-100:]
        self.pitch_line.set_data(range(len(self.pitch_history)), self.pitch_history)
        self.ax_pitch.relim()
        self.ax_pitch.autoscale_view()

        self.amplitude_history.append(amplitude)
        if len(self.amplitude_history) > 100:
            self.amplitude_history = self.amplitude_history[-100:]
        
        for i in range(2):
            if len(formants) > i:
                f_val = formants[i]
            else:
                f_val = np.nan
            self.formant_histories[i].append(f_val)
            if len(self.formant_histories[i]) > 100:
                self.formant_histories[i] = self.formant_histories[i][-100:]
            self.formant_lines[i].set_data(range(len(self.formant_histories[i])), self.formant_histories[i])
        self.ax_formants.relim()
        self.ax_formants.autoscale_view()
        self.canvas.draw_idle()

        # Compute quality metrics continuously and update quality label
        periods = [1.0 / p for p in self.pitch_history if p and p > 0 and not np.isnan(p)]
        jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100 if len(periods) >= 2 else np.nan
        valid_amps = [a for a in self.amplitude_history if a is not None and not np.isnan(a)]
        shimmer = np.mean(np.abs(np.diff(valid_amps))) / np.mean(valid_amps) * 100 if len(valid_amps) >= 2 else np.nan
        quality_text = f"Jitter: {jitter:.2f}% | Shimmer: {shimmer:.2f}% | HNR: {hnr:.2f} dB"
        self.quality_label.config(text=quality_text)

        # Also accumulate data for the metrics summary (do not affect live charts)
        if pitch is not None:
            self.metrics_pitch_history.append(pitch)
        for i in range(2):
            if len(formants) > i:
                self.metrics_formant_histories[i].append(formants[i])
            else:
                self.metrics_formant_histories[i].append(np.nan)

    def update_callback(self, note, pitch, formants, hnr, amplitude):
        self.root.after(0, self.update_ui, note, pitch, formants, hnr, amplitude)

    def get_sensitivity(self):
        return self.sensitivity_value.get()
        
    def start_listening(self):
        selected_name = self.selected_mic.get()
        device_index = self.mic_mapping.get(selected_name)
        if device_index is None:
            print("Selected microphone not found!")
            return
        pa = pyaudio.PyAudio()
        try:
            info = pa.get_device_info_by_index(device_index)
            device_rate = int(info.get('defaultSampleRate', 44100))
        except Exception as e:
            print("Error getting device info:", e)
            device_rate = 44100
        pa.terminate()
        print(f"Starting audio on '{selected_name}' (device {device_index}) at {device_rate} Hz.")
        self.audio_thread = AudioThread(self.update_callback, device_index, device_rate, self.get_sensitivity)
        self.audio_thread.start()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
    def stop_listening(self):
        if self.audio_thread and self.audio_thread.running:
            self.audio_thread.stop()
            self.audio_thread.join()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.tone_label.config(text="N/A")
            
    def reset_interval_metrics(self):
        # Compute summary metrics from the accumulated interval data
        if self.metrics_pitch_history:
            pitch_array = np.array(self.metrics_pitch_history)
            pitch_min = np.nanmin(pitch_array)
            pitch_max = np.nanmax(pitch_array)
            pitch_avg = np.nanmean(pitch_array)
            pitch_text = f"Low: {pitch_min:.1f} Hz | High: {pitch_max:.1f} Hz | Avg: {pitch_avg:.1f} Hz"
        else:
            pitch_text = "Low: N/A | High: N/A | Avg: N/A"
        formant_summary_text = ""
        for i in range(2):
            channel = np.array(self.metrics_formant_histories[i])
            if len(channel) == 0 or np.all(np.isnan(channel)):
                summary = f"F{i+1}: Low: N/A, High: N/A, Avg: N/A   "
            else:
                low = np.nanmin(channel)
                high = np.nanmax(channel)
                avg = np.nanmean(channel)
                summary = f"F{i+1}: Low: {low:.1f} Hz, High: {high:.1f} Hz, Avg: {avg:.1f} Hz   "
            formant_summary_text += summary
        self.pitch_summary_label.config(text="Pitch Summary: " + pitch_text)
        self.formant_summary_label.config(text="Formant Summary: " + formant_summary_text)
        # Clear only the metrics data (leave the live chart history intact)
        self.metrics_pitch_history = []
        self.metrics_formant_histories = [[], []]

    def schedule_reset(self):
        try:
            interval = float(self.reset_interval.get())
        except Exception:
            interval = 3.0
        self.reset_interval_metrics()
        self.reset_timer = self.root.after(int(interval * 1000), self.schedule_reset)

    def on_close(self):
        if self.reset_timer is not None:
            self.root.after_cancel(self.reset_timer)
        if self.audio_thread and self.audio_thread.running:
            self.audio_thread.stop()
            self.audio_thread.join()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
