# advanced_voice_chat.py
# Program de chat vocal cu Gemini AI - Sistem Audio Avansat (CU STREAMING TTS și AUTO-CALIBRARE)

import sys
import os
import json
import time
import threading
import queue
import asyncio
import re
from datetime import datetime
import warnings
import tempfile
import wave
import collections
import random

# ... (Sistemul de logging rămâne neschimbat) ...
LOG_CONFIG = {
    "app": True, "config": True, "cleanup": True, "audio": False, "vad": True,
    "process": True, "transcription": True, "voice": True, "tts": True,
    "tts_debug": False, "echo": True, "mute": True, "gemini": True,
    "gemini_debug": True, "semafor": False,
}
START_TIME = time.time()
def log_timestamp(message, category="app"):
    if LOG_CONFIG.get(category, True):
        elapsed = time.time() - START_TIME
        print(f"[{elapsed:8.3f}s] {message}")


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QTextEdit, QGroupBox, QFormLayout,
                               QSlider, QMessageBox, QCheckBox, QTabWidget, QSpinBox, QDialog,
                               QDialogButtonBox, QComboBox)
from PySide6.QtCore import QThread, Signal, QObject, Qt, QTimer, Slot
from PySide6.QtGui import QColor, QFont, QScreen, QTextCursor
import google.generativeai as genai
from dotenv import load_dotenv
import edge_tts
import pygame
import speech_recognition as sr
import torch
import sounddevice as sd
import numpy as np
from PIL import ImageGrab

load_dotenv()


# ADAUGĂ IMPORT ȘI VERIFICARE PENTRU LIBRĂRIA MARKDOWN
try:
    import markdown
except ImportError:
    QMessageBox.critical(None, "Librărie Lipsă", "Te rog instalează librăria 'markdown' folosind comanda: pip install markdown")
    sys.exit(1)

# =================================================================================
# ⭐ FUNCȚIE NOUĂ PENTRU CURĂȚAREA FIȘIERELOR TEMPORARE
# =================================================================================
def cleanup_temp_files():
    """Șterge fișierele temp_speech... orfane din folderul rădăcină."""
    log_timestamp("🧹 [CLEANUP] Se caută fișiere temporare vechi la pornire...", "cleanup")
    deleted_count = 0
    current_dir = os.getcwd()
    
    for filename in os.listdir(current_dir):
        # Verificăm dacă fișierul corespunde EXACT formatului nostru
        if filename.startswith("temp_speech_") and filename.endswith(".mp3"):
            full_path = os.path.join(current_dir, filename)
            if os.path.isfile(full_path):
                try:
                    os.remove(full_path)
                    log_timestamp(f"  -> Șters: {filename}", "cleanup")
                    deleted_count += 1
                except Exception as e:
                    log_timestamp(f"  -> ⚠️ Eroare la ștergerea {filename}: {e}", "cleanup")
    
    if deleted_count > 0:
        log_timestamp(f"✅ [CLEANUP] Curățenie finalizată. {deleted_count} fișiere șterse.", "cleanup")
    else:
        log_timestamp("✅ [CLEANUP] Niciun fișier temporar de șters.", "cleanup")


# ... (Clasa ContinuousVoiceWorker rămâne neschimbată) ...
class ContinuousVoiceWorker(QObject):
    """Worker pentru ascultare continuă cu Silero VAD (din main_app.py)"""
    
    language_lock_requested = Signal(str)
    speech_activity_changed = Signal(bool)
    pause_progress_updated = Signal(int)
    speech_time_updated = Signal(float)
    speech_timeout = Signal()
    
    transcription_ready = Signal(str)
    status_changed = Signal(str)
    calibration_done = Signal(float)
    audio_level_changed = Signal(float)
    speaker_identified = Signal(str, float)
    
    def __init__(self, threshold, pause_duration, margin_percent, max_speech_duration, enable_echo_cancellation, vad_model):
        super().__init__()
        self._is_running = False
        self._is_muted = False
        self.enable_echo_cancellation = enable_echo_cancellation
        self.enable_speaker_identification = False
        
        # --- BLOC MODIFICAT ---
        # Nu mai încărcăm modelul aici, îl primim gata încărcat
        self.vad_model = vad_model
        log_timestamp("🎤 [VAD INIT] Model VAD pre-încărcat a fost primit.", "vad")
        # --- SFÂRȘIT BLOC MODIFICAT ---
        
        self.current_lock_mode = 'auto'
        self.primary_language = "ro-RO"
        self.secondary_language = "ro-RO"
        self.sample_rate = 16000
        self.frame_duration = 32
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.threshold = threshold
        self.pause_duration = pause_duration
        self.margin_percent = margin_percent
        self.max_speech_duration = max_speech_duration
        self.speech_threshold = 0.5
        self.silence_threshold = 0.3
        self.silence_frames_threshold = int((self.pause_duration * 1000) / self.frame_duration)
        self.MAX_SPEECH_FRAMES = int(self.max_speech_duration * 1000 / self.frame_duration)
        self.ring_buffer_size = int(self.sample_rate * 0.5)
        self.ring_buffer = collections.deque(maxlen=self.ring_buffer_size // self.frame_size)
        self.is_speech_active = False
        self.frames_since_silence = 0
        self.speech_frames = []
        self.last_ai_text = ""
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = threshold
        log_timestamp("🎤 [VAD INIT] Silero VAD inițializat", "vad")

    def set_primary_language(self, lang_code):
        if self.primary_language != lang_code:
            self.primary_language = lang_code
            log_timestamp(f"🗣️ [TRANSCRIERE] Limba primară setată la: '{lang_code}'", "transcription")

    def set_last_ai_text(self, text):
        self.last_ai_text = text
        log_timestamp(f"🔊 [ECHO PROTECTION] Salvat text AI: '{text[:50]}...'", "echo")

    def set_muted(self, muted, is_ai_speaking=True):
        self._is_muted = muted
        if not muted:
            self.ring_buffer.clear()
            self.speech_frames = []
            self.is_speech_active = False
            log_timestamp("🗑️ [MUTING] Buffer-ul audio golit la unmute", "mute")
        
        if muted:
            if is_ai_speaking:
                log_timestamp("🔇 [MUTING] Ascultare PAUSATĂ (AI vorbește)", "mute")
                self.status_changed.emit("🔇 Pausat (AI vorbește)")
            else:
                log_timestamp("🔇 [MUTING] Ascultare PAUSATĂ", "mute")
                self.status_changed.emit("🎧 Mut")
        else:
            log_timestamp("🔊 [MUTING] Ascultare RELUATĂ", "mute")
            self.status_changed.emit("⚪ Aștept să vorbești...")

    def set_max_speech_duration(self, seconds):
        self.max_speech_duration = seconds
        self.MAX_SPEECH_FRAMES = int(seconds * 1000 / self.frame_duration)
        log_timestamp(f"🎤 [WORKER UPDATE] Durata maximă setată la {seconds}s.", "vad")

    def is_echo(self, transcribed_text):
        if not self.enable_echo_cancellation: return False
        if not self.last_ai_text or not transcribed_text: return False
        ai_normalized = ''.join(c for c in self.last_ai_text.lower() if c.isalnum() or c.isspace())
        transcribed_normalized = ''.join(c for c in transcribed_text.lower() if c.isalnum() or c.isspace())
        ai_words = set(ai_normalized.split())
        transcribed_words = transcribed_normalized.split()
        if len(transcribed_words) == 0: return False
        common_words = sum(1 for word in transcribed_words if word in ai_words)
        similarity = common_words / len(transcribed_words)
        is_echo_detected = similarity > 0.75
        if is_echo_detected: log_timestamp(f"🚫 [ECHO DETECTAT] '{transcribed_text}'", "echo")
        return is_echo_detected

    def audio_callback(self, indata, frames, time_info, status):
        if status: log_timestamp(f"⚠️ [AUDIO] Status: {status}", "audio")
        audio_data = indata[:, 0].copy()
        rms = np.sqrt(np.mean(audio_data.astype(float)**2))
        if rms > 0:
            db_level = 20 * np.log10(rms) + 90
            self.audio_level_changed.emit(min(max(db_level * 50, 0), 10000))
        if self._is_muted: return
        audio_tensor = torch.from_numpy(audio_data).float()
        with torch.no_grad():
            speech_probability = self.vad_model(audio_tensor, self.sample_rate).item()
        is_speech = speech_probability > self.speech_threshold
        audio_int16 = (audio_data * 32767).astype(np.int16)
        self.ring_buffer.append(audio_int16)
        if is_speech:
            if not self.is_speech_active:
                self.is_speech_active = True
                self.speech_activity_changed.emit(True)
                self.pause_progress_updated.emit(100)
                log_timestamp("🟢 [VAD] Început vorbire detectat", "vad")
                self.frames_since_silence = 0
                self.speech_frames = list(self.ring_buffer)
                self.status_changed.emit("🔵 Vorbești...")
            else:
                self.frames_since_silence = 0
                self.speech_frames.append(audio_int16)
                self.pause_progress_updated.emit(100)
        else:
            if self.is_speech_active:
                self.frames_since_silence += 1
                self.speech_frames.append(audio_int16)
                progress = 100 - int(100 * self.frames_since_silence / self.silence_frames_threshold)
                self.pause_progress_updated.emit(progress)
        if self.is_speech_active:
            timp_ramas = (self.MAX_SPEECH_FRAMES - len(self.speech_frames)) * self.frame_duration / 1000.0
            self.speech_time_updated.emit(timp_ramas)
        should_process_due_to_pause = self.is_speech_active and self.frames_since_silence >= self.silence_frames_threshold
        should_process_due_to_length = self.is_speech_active and len(self.speech_frames) >= self.MAX_SPEECH_FRAMES
        if should_process_due_to_pause or should_process_due_to_length:
            if should_process_due_to_length:
                log_timestamp("🔴 [VAD] Limita de timp atinsă! Procesare forțată.", "vad")
                self.speech_timeout.emit()
            else:
                log_timestamp(f"🔴 [VAD] Sfârșit vorbire (pauză).", "vad")
                self.speech_activity_changed.emit(False)
            self.speech_time_updated.emit(-1)
            self.process_captured_speech()
            self.is_speech_active = False
            self.frames_since_silence = 0
            self.speech_frames = []

    def process_captured_speech(self):
        if len(self.speech_frames) == 0: return
        temp_path = None
        try:
            audio_data = np.concatenate(self.speech_frames)
            duration = len(audio_data) / self.sample_rate
            if duration < 0.3:
                self.status_changed.emit("⚪ Aștept să vorbești...")
                return
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_path = temp_wav.name
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data.tobytes())
            with sr.AudioFile(temp_path) as source:
                audio = self.recognizer.record(source)
            self.status_changed.emit("🟡 Transcriu...")
            text = None
            try:
                text = self.recognizer.recognize_google(audio, language=self.primary_language)
            except sr.UnknownValueError:
                self.status_changed.emit("⚠️ Nu am înțeles")
                return
            except sr.RequestError as e:
                self.status_changed.emit(f"⚠️ Eroare API: {e}")
                return
            if text:
                if self.is_echo(text):
                    self.status_changed.emit("⚪ Aștept să vorbești...")
                    return
                self.transcription_ready.emit(text)
        except Exception as e:
            log_timestamp(f"❌ [PROCESS] Eroare în procesarea audio: {e}", "process")
            self.status_changed.emit("⚠️ Eroare procesare")
        finally:
            if temp_path and os.path.exists(temp_path):
                try: os.unlink(temp_path)
                except Exception: pass

    def run(self):
        log_timestamp("🎤 [SILERO VAD WORKER] Worker pornit", "vad")
        self._is_running = True
        self.status_changed.emit("⚪ Aștept să vorbești...")
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', blocksize=self.frame_size, callback=self.audio_callback):
                log_timestamp("✅ [SILERO VAD WORKER] Stream audio pornit", "vad")
                while self._is_running:
                    sd.sleep(100)
        except Exception as e:
            log_timestamp(f"❌ [SILERO VAD WORKER] EROARE CRITICĂ: {e}", "vad")
            self.status_changed.emit(f"⚠️ Eroare: {e}")
        finally:
            log_timestamp("🎤 [SILERO VAD WORKER] Worker oprit", "vad")

    def stop(self):
        self._is_running = False

# ... (Clasa StreamingTTSManager rămâne neschimbată) ...
class StreamingTTSSignals(QObject):
    all_sentences_finished = Signal()
    error_occurred = Signal(str)
    play_audio_file = Signal(str)

class StreamingTTSManager:
    def __init__(self):
        self.signals = StreamingTTSSignals()
        self.tts_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.is_generating = False
        self.is_playing = False
        self._stop_requested = False
        self.generator_thread = None
        self.player_thread = None
        self.current_voice = "ro-RO-EmilNeural"
        self._playback_finished_event = None
        log_timestamp("🔊 [STREAMING TTS] Manager inițializat", "tts")

    def start_speaking(self, text, voice_id):
        if self.is_generating:
            self.stop_all()
            time.sleep(0.3)
        self.current_voice = voice_id
        self._stop_requested = False
        sentences = self._split_into_sentences(text)
        for sentence in sentences: self.tts_queue.put(sentence)
        self.tts_queue.put(None)
        self._start_generator_worker()
        self._start_player_worker()

    def _split_into_sentences(self, text):
        clean_text = re.sub(r'\[EMOTION:\w+\]\s*', '', text)
        sentences = []
        current = ""
        for char in clean_text:
            current += char
            if char in '.!?':
                if current.strip(): sentences.append(current.strip())
                current = ""
        if current.strip(): sentences.append(current.strip())
        return sentences if sentences else [clean_text]

    def _start_generator_worker(self):
        if self.generator_thread and self.generator_thread.is_alive(): return
        self.is_generating = True
        self.generator_thread = threading.Thread(target=self._generator_worker, daemon=True)
        self.generator_thread.start()

    def _start_player_worker(self):
        if self.player_thread and self.player_thread.is_alive(): return
        self.is_playing = True
        self.player_thread = threading.Thread(target=self._player_worker, daemon=True)
        self.player_thread.start()

    def _generator_worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while not self._stop_requested:
                text_chunk = self.tts_queue.get()
                if text_chunk is None: break
                if text_chunk.strip(): loop.run_until_complete(self._generate_audio_file(text_chunk))
                self.tts_queue.task_done()
        except Exception as e:
            self.signals.error_occurred.emit(str(e))
        finally:
            self.audio_queue.put(None)
            self.is_generating = False

    async def _generate_audio_file(self, text):
        output_file = f"temp_speech_{int(time.time()*1000)}_{random.randint(1000,9999)}.mp3"
        try:
            communicate = edge_tts.Communicate(text, self.current_voice)
            await communicate.save(output_file)
            self.audio_queue.put(output_file)
        except Exception as e:
            if os.path.exists(output_file): os.remove(output_file)
            raise

    def _player_worker(self):
        try:
            while not self._stop_requested:
                audio_path = self.audio_queue.get()
                if audio_path is None: break
                self._playback_finished_event = threading.Event()
                self.signals.play_audio_file.emit(audio_path)
                self._playback_finished_event.wait()
                if os.path.exists(audio_path):
                    try: os.remove(audio_path)
                    except Exception: pass
                self.audio_queue.task_done()
        except Exception as e:
            self.signals.error_occurred.emit(str(e))
        finally:
            self.is_playing = False
            self.signals.all_sentences_finished.emit()

    def stop_all(self):
        self._stop_requested = True
        try:
            pygame.mixer.music.stop(); pygame.mixer.music.unload()
        except: pass
        if self._playback_finished_event and not self._playback_finished_event.is_set():
            self._playback_finished_event.set()
        while not self.tts_queue.empty():
            try: self.tts_queue.get_nowait()
            except: break
        while not self.audio_queue.empty():
            try:
                item = self.audio_queue.get_nowait()
                if item and os.path.exists(item): os.remove(item)
            except: break
        if self.generator_thread and self.generator_thread.is_alive(): self.generator_thread.join(timeout=1.0)
        if self.player_thread and self.player_thread.is_alive(): self.player_thread.join(timeout=1.0)
        self.is_generating = self.is_playing = False


class AdvancedVoiceChatApp(QWidget):
    gemini_response_signal = Signal(str)
    CONFIG_FILE = "voice_chat_config.json"
    
    def load_config(self):
        # --- MODIFICAT: Adăugăm noua setare de auto-calibrare ---
        default_config = {
            "threshold": 4000, "pause_duration": 1.5, "max_speech_duration": 15,
            "enable_echo_cancellation": True, "tts_enabled": True,
            "selected_voice": "ro-RO-EmilNeural",
            "custom_system_prompt": "Ești un asistent util și prietenos. Răspunde concis și clar în limba română.",
            "conversation_memory_limit": 10,
            "auto_calibrate_on_start": True,  # <-- SETARE AUTO-CALIBRARE
            "desktop_assistant_mode": False,  # <-- SETARE DESKTOP ASSISTANT
            "selected_model": "gemini-flash-latest"  # <-- MODEL AI SELECTAT
        }
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                log_timestamp("✅ [CONFIG] Configurație încărcată din fișier.", "config")
        except Exception as e:
            log_timestamp(f"⚠️ [CONFIG] Eroare la încărcarea configurației: {e}. Se folosesc valori implicite.", "config")
        
        self.voice_config["threshold"] = default_config["threshold"]
        self.voice_config["pause_duration"] = default_config["pause_duration"]
        self.voice_config["max_speech_duration"] = default_config["max_speech_duration"]
        self.voice_config["enable_echo_cancellation"] = default_config["enable_echo_cancellation"]
        self.tts_enabled = default_config["tts_enabled"]
        self.selected_voice = default_config["selected_voice"]
        self.custom_system_prompt = default_config["custom_system_prompt"]  # Valoare implicit
        self.conversation_memory_limit = default_config["conversation_memory_limit"]
        self.auto_calibrate_on_start = default_config["auto_calibrate_on_start"] # <-- ÎNCĂRCĂM SETAREA
        self.desktop_assistant_mode = default_config["desktop_assistant_mode"] # <-- ÎNCĂRCĂM DESKTOP ASSISTANT
        self.selected_model = default_config["selected_model"]  # <-- ÎNCĂRCĂM MODELUL
        log_timestamp(f"⚙️ [CONFIG] Auto-calibrare la pornire încărcat: {self.auto_calibrate_on_start}", "config")
        log_timestamp(f"⚙️ [CONFIG] Desktop Assistant Mode încărcat: {self.desktop_assistant_mode}", "config")
        log_timestamp(f"🤖 [CONFIG] Model AI încărcat: {self.selected_model}", "config")
    
    # --- FUNCȚII NOI PENTRU PROMPT EXTERN ---
    PROMPT_FILE = "system_prompt.txt"
    
    def load_system_prompt(self):
        """Încarcă prompt-ul de sistem din fișierul extern."""
        try:
            if os.path.exists(self.PROMPT_FILE):
                with open(self.PROMPT_FILE, 'r', encoding='utf-8') as f:
                    prompt_text = f.read().strip()
                    if prompt_text:
                        self.custom_system_prompt = prompt_text
                        log_timestamp(f"📄 [PROMPT] Prompt încărcat din {self.PROMPT_FILE} ({len(prompt_text)} caractere)", "config")
                        return True
                    else:
                        log_timestamp(f"⚠️ [PROMPT] Fișier {self.PROMPT_FILE} este gol, se folosește prompt-ul implicit.", "config")
            else:
                log_timestamp(f"ℹ️ [PROMPT] Fișier {self.PROMPT_FILE} nu există, se creează cu prompt-ul implicit.", "config")
                self.save_system_prompt()
        except Exception as e:
            log_timestamp(f"❌ [PROMPT] Eroare la încărcarea prompt-ului: {e}", "config")
        return False
    
    def save_system_prompt(self):
        """Salvează prompt-ul de sistem în fișierul extern."""
        try:
            with open(self.PROMPT_FILE, 'w', encoding='utf-8') as f:
                f.write(self.custom_system_prompt)
            log_timestamp(f"💾 [PROMPT] Prompt salvat în {self.PROMPT_FILE} ({len(self.custom_system_prompt)} caractere)", "config")
            return True
        except Exception as e:
            log_timestamp(f"❌ [PROMPT] Eroare la salvarea prompt-ului: {e}", "config")
            return False
    
    def reload_system_prompt(self):
        """Reîncarcă prompt-ul din fișier și reinițializează modelul."""
        if self.load_system_prompt():
            # Reinițializăm modelul cu noul prompt (folosim modelul selectat)
            self.model = genai.GenerativeModel(model_name=self.selected_model, system_instruction=self.custom_system_prompt)
            self.chat = self.model.start_chat(history=[])
            self.conversation_history = []
            
            # Actualizăm preview-ul în interfață
            preview_text = self.custom_system_prompt[:100] + "..." if len(self.custom_system_prompt) > 100 else self.custom_system_prompt
            self.prompt_preview.setText(f"Prompt actual: {preview_text}")
            
            log_timestamp("🔄 [PROMPT] Prompt reîncărcat și model reinițializat!", "config")
            QMessageBox.information(self, "Succes", f"Prompt-ul a fost reîncărcat din {self.PROMPT_FILE}!\nConversația a fost resetată.")
        else:
            QMessageBox.warning(self, "Eroare", f"Nu s-a putut reîncărca prompt-ul din {self.PROMPT_FILE}")
    # --- SFÂRȘIT FUNCȚII NOI ---

    def save_config(self):
        config = {
            "threshold": self.voice_config["threshold"],
            "pause_duration": self.voice_config["pause_duration"],
            "max_speech_duration": self.voice_config["max_speech_duration"],
            "enable_echo_cancellation": self.voice_config["enable_echo_cancellation"],
            "tts_enabled": self.tts_enabled,
            "selected_voice": self.selected_voice,
            # custom_system_prompt NU mai e salvat aici - se salvează în system_prompt.txt
            "conversation_memory_limit": self.conversation_memory_limit,
            "auto_calibrate_on_start": self.auto_calibrate_on_start, # <-- SALVĂM AUTO-CALIBRARE
            "desktop_assistant_mode": self.desktop_assistant_mode,  # <-- SALVĂM DESKTOP ASSISTANT
            "selected_model": self.selected_model  # <-- SALVĂM MODELUL AI
        }
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            log_timestamp(f"💾 [CONFIG] Salvat: model={self.selected_model}, auto_calibrate={self.auto_calibrate_on_start}, desktop_mode={self.desktop_assistant_mode}", "config")
        except Exception as e:
            log_timestamp(f"❌ [CONFIG] Eroare la salvarea configurației: {e}", "config")
    
    def __init__(self):
        super().__init__()
        
        # Verificare API Key (neschimbat)
        api_key = os.getenv("GOOGLE_API_KEY") or self._prompt_for_api_key()[0]
        if not api_key:
            QMessageBox.critical(self, "Eroare", "Cheia API Google Gemini este obligatorie!")
            sys.exit(1)
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            QMessageBox.critical(self, "Eroare", f"Cheia API nu este validă: {e}")
            sys.exit(1)
            
        # Încărcarea modelului VAD la pornire (neschimbat)
        log_timestamp("🧠 [APP INIT] Se încarcă modelul Silero VAD (o singură dată)...", "app")
        try:
            torch.set_num_threads(1)
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', model='silero_vad',
                force_reload=False, onnx=False)
        except Exception as e:
            QMessageBox.critical(self, "Eroare Critică", f"Nu s-a putut încărca modelul de detecție vocală:\n{e}\nAplicația se va închide.")
            sys.exit(1)
            
        pygame.mixer.init()
        self.streaming_tts = StreamingTTSManager()
        
        # --- BLOC NOU ---
        self.desktop_assistant_mode = False
        os.makedirs("screenshots", exist_ok=True) # Creează folderul dacă nu există
        # --- SFÂRȘIT BLOC NOU ---

        # --- MODELE AI DISPONIBILE ---
        self.available_models = {
            "Gemini Flash (Rapid)": "gemini-flash-latest",
            "Gemini Pro (Avansat)": "gemini-pro-latest"
        }
        # --- SFÂRȘIT MODELE ---

        self.romanian_voices = {"Emil (Bărbat)": "ro-RO-EmilNeural", "Alina (Femeie)": "ro-RO-AlinaNeural"}
        self.voice_config = {"margin_percent": 25}
        self.load_config()
        
        # --- ÎNCĂRCĂM PROMPT-UL DIN FIȘIER EXTERN ---
        self.load_system_prompt()
        # --- SFÂRȘIT ÎNCĂRCARE PROMPT ---
        
        # Folosim modelul selectat din config
        self.model = genai.GenerativeModel(model_name=self.selected_model, system_instruction=self.custom_system_prompt)
        self.chat = self.model.start_chat(history=[])
        self.conversation_history = []
        self.voice_enabled = self.is_muted = False
        self.voice_worker = self.voice_thread = None
        self.gemini_response_signal.connect(self.display_gemini_response)
        self.streaming_tts.signals.all_sentences_finished.connect(self.on_all_sentences_finished)
        self.streaming_tts.signals.error_occurred.connect(self.on_streaming_tts_error)
        self.streaming_tts.signals.play_audio_file.connect(self.on_play_audio_file)
        self.pygame_check_timer = QTimer(self)
        self.pygame_check_timer.timeout.connect(self._check_pygame_playback)
        self.init_ui()
        
        # Actualizăm preview-ul prompt-ului după ce UI-ul e creat
        preview_text = self.custom_system_prompt[:100] + "..." if len(self.custom_system_prompt) > 100 else self.custom_system_prompt
        self.prompt_preview.setText(f"Prompt actual: {preview_text}")

    def create_audio_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        controls_group = QGroupBox("🎛️ Controale Audio")
        controls_layout = QFormLayout()

        self.auto_calibrate_checkbox = QCheckBox("Calibrează automat la pornire (recomandat)")
        self.auto_calibrate_checkbox.setChecked(self.auto_calibrate_on_start)
        self.auto_calibrate_checkbox.stateChanged.connect(self.on_auto_calibrate_changed)
        controls_layout.addRow(self.auto_calibrate_checkbox)
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        
        # --- MODIFICAT AICI ---
        self.threshold_slider.setRange(100, 12000)
        # --- SFÂRȘIT MODIFICARE ---
        
        self.threshold_slider.setValue(self.voice_config["threshold"])
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.threshold_label = QLabel(f"{self.voice_config['threshold']}")
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        controls_layout.addRow("Prag Energie (manual):", threshold_layout)
        
        # ... restul funcției rămâne neschimbat ...
        self.pause_slider = QSlider(Qt.Orientation.Horizontal)
        self.pause_slider.setRange(5, 50)
        self.pause_slider.setValue(int(self.voice_config["pause_duration"] * 10))
        self.pause_slider.valueChanged.connect(self.on_pause_changed)
        self.pause_label = QLabel(f"{self.voice_config['pause_duration']:.1f}s")
        pause_layout = QHBoxLayout()
        pause_layout.addWidget(self.pause_slider)
        pause_layout.addWidget(self.pause_label)
        controls_layout.addRow("Pauză Sfârșit:", pause_layout)
        
        self.max_speech_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_speech_slider.setRange(5, 30)
        self.max_speech_slider.setValue(self.voice_config["max_speech_duration"])
        self.max_speech_slider.valueChanged.connect(self.on_max_speech_changed)
        self.max_speech_label = QLabel(f"{self.voice_config['max_speech_duration']}s")
        max_speech_layout = QHBoxLayout()
        max_speech_layout.addWidget(self.max_speech_slider)
        max_speech_layout.addWidget(self.max_speech_label)
        controls_layout.addRow("Durată Max Vorbire:", max_speech_layout)
        
        self.echo_checkbox = QCheckBox("Activat")
        self.echo_checkbox.setChecked(self.voice_config["enable_echo_cancellation"])
        self.echo_checkbox.stateChanged.connect(self.on_echo_changed)
        controls_layout.addRow("Anulare Ecou:", self.echo_checkbox)
        
        self.tts_checkbox = QCheckBox("Activat")
        self.tts_checkbox.setChecked(self.tts_enabled)
        self.tts_checkbox.stateChanged.connect(self.on_tts_changed)
        controls_layout.addRow("Text-to-Speech (TTS):", self.tts_checkbox)
        
        self.voice_combo = QComboBox()
        for voice_name in self.romanian_voices.keys():
            self.voice_combo.addItem(voice_name)
        for idx, (name, code) in enumerate(self.romanian_voices.items()):
            if code == self.selected_voice:
                self.voice_combo.setCurrentIndex(idx)
                break
        self.voice_combo.currentTextChanged.connect(self.on_voice_changed)
        controls_layout.addRow("Voce TTS Română:", self.voice_combo)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    # --- FUNCȚIE NOUĂ: Handler pentru checkbox ---
    def on_auto_calibrate_changed(self, state):
        # Același fix ca la Desktop Assistant - comparăm cu valoarea integer
        self.auto_calibrate_on_start = (state == Qt.CheckState.Checked.value) or (state == 2)
        log_timestamp(f"⚙️ [CONFIG] Calibrare automată setată la: {self.auto_calibrate_on_start}", "config")
        self.save_config()

    # --- FUNCȚIE NOUĂ: Logica de calibrare ---
    def _run_auto_calibration(self):
        log_timestamp("🤫 [CALIBRARE] Se calibrează pragul de energie... Stai în liniște 2s.", "app")
        self.update_status("🤫 Calibrez... Liniște 2s")
        QApplication.processEvents() # Forțează actualizarea UI

        try:
            recognizer = sr.Recognizer()
            with sr.Microphone(sample_rate=16000) as source:
                recognizer.adjust_for_ambient_noise(source, duration=2)
            
            noise_level = recognizer.energy_threshold
            # Folosim o marjă fixă de 20% peste zgomot
            new_threshold = int(noise_level * 1.20) 
            
            # --- MODIFICAT AICI ---
            # Ne asigurăm că pragul nu e prea mic sau prea mare (până la 12000)
            new_threshold = max(100, min(new_threshold, 12000))
            # --- SFÂRȘIT MODIFICARE ---

            log_timestamp(f"✅ [CALIBRARE] Zgomot: {noise_level:.0f}, Prag nou: {new_threshold}", "app")
            
            # Actualizăm valoarea în config și pe slider
            self.voice_config["threshold"] = new_threshold
            self.threshold_slider.setValue(new_threshold)
            self.update_status("✅ Calibrare finalizată!")
            QApplication.processEvents()
            time.sleep(1) # Lasă utilizatorul să vadă mesajul

        except Exception as e:
            log_timestamp(f"❌ [CALIBRARE] Eroare: {e}", "app")
            self.update_status(f"⚠️ Eroare calibrare: {e}")
            time.sleep(2)

    def toggle_voice(self):
        """Activează/dezactivează microfonul"""
        if not self.voice_enabled:
            if self.auto_calibrate_on_start:
                self._run_auto_calibration()

            self.voice_enabled = True
            self.voice_toggle_button.setText("🔴 Oprește Microfonul")
            self.voice_toggle_button.setStyleSheet("background-color: #e74c3c; font-size: 14px; padding: 10px; font-weight: bold;")
            self.mute_button.setEnabled(True)
            self.voice_thread = QThread(self)
            
            # --- MODIFICARE AICI: Pasăm modelul pre-încărcat ---
            self.voice_worker = ContinuousVoiceWorker(
                threshold=self.voice_config["threshold"], 
                pause_duration=self.voice_config["pause_duration"],
                margin_percent=self.voice_config["margin_percent"], 
                max_speech_duration=self.voice_config["max_speech_duration"],
                enable_echo_cancellation=self.voice_config["enable_echo_cancellation"],
                vad_model=self.vad_model # <-- PARAMETRU NOU
            )
            
            # Atribuim și utilitarele, chiar dacă nu le folosim direct aici
            self.voice_worker.vad_utils = self.vad_utils
            # --- SFÂRȘIT MODIFICARE ---

            self.voice_worker.moveToThread(self.voice_thread)
            self.voice_worker.transcription_ready.connect(self.on_transcription_ready)
            self.voice_worker.status_changed.connect(self.update_status)
            self.voice_worker.speech_activity_changed.connect(self.on_speech_activity_changed)
            self.voice_worker.pause_progress_updated.connect(self.on_pause_progress_updated)
            self.voice_worker.speech_time_updated.connect(self.on_speech_time_updated)
            self.voice_worker.speech_timeout.connect(self.on_speech_timeout)
            self.voice_thread.started.connect(self.voice_worker.run)
            self.voice_thread.start()
        else:
            self.voice_enabled = False
            self.voice_toggle_button.setText("🟢 Activează Microfonul")
            self.voice_toggle_button.setStyleSheet("background-color: #27ae60; font-size: 14px; padding: 10px; font-weight: bold;")
            self.mute_button.setEnabled(False)
            self.is_muted = False
            if self.voice_worker: self.voice_worker.stop()
            if self.voice_thread:
                self.voice_thread.quit()
                self.voice_thread.wait()
            self.update_status("Gata de conversație")
            self._update_semafor("rosu")
    
    
    def get_gemini_response(self, text):
        """Obține răspuns de la Gemini, cu sau fără screenshot, în funcție de mod."""
        QTimer.singleShot(0, lambda: self.update_status("⏳ Aștept răspunsul..."))
        if self.voice_worker:
            self.voice_worker.set_muted(True, is_ai_speaking=True)
            
        try:
            full_response = ""
            
            # ADĂUGĂM textul user în istoric ÎNAINTE de request (comun pentru ambele moduri)
            self.conversation_history.append({"role": "user", "parts": [text]})
            log_timestamp(f"💾 [ISTORIC] Mesaj user adăugat (total: {len(self.conversation_history)} mesaje)", "gemini")
            
            # Tăiem istoricul dacă e prea lung
            if len(self.conversation_history) > self.conversation_memory_limit * 2:
                self.conversation_history = self.conversation_history[-(self.conversation_memory_limit * 2):]
                log_timestamp(f"✂️ [ISTORIC] Tăiat la {self.conversation_memory_limit * 2} mesaje", "gemini")
            
            if self.desktop_assistant_mode:
                # --- MODUL ASISTENT DESKTOP (CU SCREENSHOT) ---
                log_timestamp("=" * 60, "app")
                log_timestamp("🤖 [ASSISTANT] MODUL ASISTENT DESKTOP ACTIVAT", "app")
                log_timestamp("=" * 60, "app")
                log_timestamp(f"📝 [ASSISTANT] Text user: '{text}'", "app")
                log_timestamp("🖼️ [ASSISTANT] Capturez ecranul...", "app")
                
                try:
                    # Preluarea geometriei monitorului principal
                    screen = QApplication.primaryScreen()
                    geometry = screen.geometry()
                    x, y, width, height = geometry.getRect()
                    log_timestamp(f"📐 [ASSISTANT] Dimensiuni: {width}x{height} @ ({x}, {y})", "app")
                    
                    # Capturarea screenshot-ului
                    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
                    log_timestamp(f"✅ [ASSISTANT] Screenshot capturat! Size: {screenshot.size}", "app")
                    
                    # Salvarea screenshot-ului
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                    filename = f"screenshot_{timestamp}.png"
                    save_path = os.path.join("screenshots", filename)
                    screenshot.save(save_path)
                    log_timestamp(f"💾 [ASSISTANT] Salvat: {save_path}", "app")
                    
                    # Creăm model viziune - FOLOSIM MODELUL SELECTAT ȘI SYSTEM PROMPT-UL
                    # ⭐ ATENȚIE: Aici este cheia! Folosim system_instruction din self.custom_system_prompt
                    vision_model = genai.GenerativeModel(
                        model_name=self.selected_model,
                        system_instruction=self.custom_system_prompt  # ⭐ ADAUGĂ ACEASTĂ LINIE!
                    )
                    model_name = "Flash" if "flash" in self.selected_model.lower() else "Pro"
                    log_timestamp(f"🤖 [ASSISTANT] Model Gemini {model_name} (viziune) init cu system prompt", "app")
                    
                    # Creăm chat cu istoric TEXT-ONLY
                    chat_with_history = vision_model.start_chat(history=self.conversation_history[:-1])
                    log_timestamp(f"📚 [ASSISTANT] Chat cu {len(self.conversation_history)-1} mesaje istoric (text-only)", "gemini")
                    
                    # ⭐⭐⭐ MODIFICAREA CRITICĂ - NU mai adăugăm instrucțiuni forțate!
                    # Trimitem textul EXACT așa cum este, fără instrucțiuni suplimentare
                    # System prompt-ul se va ocupa de când să analizeze screenshot-ul
                    prompt_text = text  # ⭐ SIMPLIFICAT! Doar textul user, fără instrucțiuni extra
                    
                    log_timestamp(f"📤 [ASSISTANT] Trimit multimodal: text + screenshot (fără instrucțiuni forțate)", "gemini")
                    
                    # Trimitem mesajul CURENT cu screenshot
                    response_stream = chat_with_history.send_message(
                        [prompt_text, screenshot],
                        stream=True
                    )
                    
                    log_timestamp("⏳ [ASSISTANT] Primesc răspuns (streaming)...", "gemini")
                    full_response = ""
                    chunk_count = 0
                    for chunk in response_stream:
                        if chunk.text:
                            full_response += chunk.text
                            chunk_count += 1
                            if chunk_count % 5 == 0:
                                log_timestamp(f"📦 [ASSISTANT] Chunk #{chunk_count}, total: {len(full_response)} chars", "gemini_debug")
                    
                    log_timestamp(f"✅ [ASSISTANT] Răspuns complet! {chunk_count} chunks, {len(full_response)} chars", "gemini")
                    log_timestamp(f"💬 [ASSISTANT] Preview: '{full_response[:150]}...'", "gemini_debug")
                    
                except Exception as screenshot_error:
                    log_timestamp(f"❌ [ASSISTANT] EROARE: {screenshot_error}", "app")
                    import traceback
                    log_timestamp(f"🔍 [ASSISTANT] Traceback:\n{traceback.format_exc()}", "gemini_debug")
                    raise screenshot_error

            else:
                # --- MODUL NORMAL (TEXT-ONLY) ---
                log_timestamp(f"🚀 [GEMINI] Modul normal (text-only)", "gemini")
                log_timestamp(f"📝 [GEMINI] Trimit: '{text}'", "gemini")
                
                self.chat = self.model.start_chat(history=self.conversation_history[:-1])
                log_timestamp(f"📚 [GEMINI] Chat cu {len(self.conversation_history)-1} mesaje istoric", "gemini")
                response_stream = self.chat.send_message(text, stream=True)
                full_response = "".join([chunk.text for chunk in response_stream if chunk.text])
                log_timestamp(f"✅ [GEMINI] Răspuns primit ({len(full_response)} chars)", "gemini")

            # --- LOGICA COMUNĂ: Salvăm răspunsul AI în istoric (DOAR TEXT) ---
            self.conversation_history.append({"role": "model", "parts": [full_response]})
            log_timestamp(f"💾 [ISTORIC] Răspuns AI salvat (total: {len(self.conversation_history)} mesaje)", "gemini")
            
            self.gemini_response_signal.emit(full_response)
            log_timestamp("📤 [GEMINI] Signal emis pentru afișare", "gemini_debug")
            
            if self.voice_worker:
                self.voice_worker.set_last_ai_text(full_response)
                log_timestamp("🔊 [ECHO] Text AI salvat pentru protecție ecou", "echo")
            
            if self.tts_enabled:
                log_timestamp("🗣️ [TTS] Pornesc TTS...", "tts")
                self.streaming_tts.start_speaking(full_response, self.selected_voice)
            else:
                log_timestamp("🔇 [TTS] TTS off, reactivare microfon", "tts")
                self.on_all_sentences_finished()
                
        except Exception as e:
            log_timestamp(f"❌ [GEMINI] EROARE CRITICĂ: {e}", "gemini")
            log_timestamp(f"📋 [GEMINI] Tip: {type(e).__name__}", "gemini")
            import traceback
            log_timestamp(f"🔍 [GEMINI] Traceback:\n{traceback.format_exc()}", "gemini_debug")
            error_msg = f"Eroare Gemini: {e}"
            QTimer.singleShot(0, lambda msg=error_msg: self.add_to_chat("Sistem", msg))
            self.on_all_sentences_finished()

    
    @Slot(str)
    def on_play_audio_file(self, audio_path):
        """Rulează în main thread pentru a reda un fișier audio cu pygame."""
        try:
            log_timestamp(f"🎵 [MAIN THREAD] Încep redare: '{audio_path}'", "tts")
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # --- LINIE NOUĂ ---
            self.stop_button.setEnabled(True) # Activăm butonul de stop
            
            self.pygame_check_timer.start(50) # Verifică la fiecare 50ms
        except Exception as e:
            log_timestamp(f"❌ [MAIN THREAD] Eroare la pornire redare: {e}", "tts")
            if self.streaming_tts._playback_finished_event:
                self.streaming_tts._playback_finished_event.set()


    def _check_pygame_playback(self):
        if not pygame.mixer.music.get_busy():
            self.pygame_check_timer.stop()
            pygame.mixer.music.unload()
            if self.streaming_tts._playback_finished_event: self.streaming_tts._playback_finished_event.set()

    @Slot()
    def on_all_sentences_finished(self):
        """Callback apelat de manager când TOATE propozițiile au fost redate."""
        log_timestamp("🏁 [STREAMING] Toate propozițiile terminate. Se reactivează microfonul.", "tts")

        # --- LINIE NOUĂ ---
        self.stop_button.setEnabled(False) # Dezactivăm butonul, nu mai are ce opri

        if self.voice_worker and not self.is_muted:
            self.voice_worker.set_muted(False, is_ai_speaking=False)
            log_timestamp("🔊 [UNMUTE] Microfon reactivat automat după TTS", "mute")
        elif self.is_muted:
            log_timestamp("🔇 [UNMUTE] Mute manual activ - NU se reactivează microfonul", "mute")

    @Slot(str)
    def on_streaming_tts_error(self, error_message):
        self.streaming_tts.stop_all()
        self.on_all_sentences_finished()
    # ... [restul codului neschimbat] ...
    
    def _prompt_for_api_key(self):
        """Deschide un dialog, cere cheia API și o salvează într-un fișier .env."""
        from PySide6.QtWidgets import QInputDialog
        
        # Am actualizat textul pentru a fi mai clar pentru utilizator
        api_key, ok = QInputDialog.getText(
            self, 
            "Cheie API Google Gemini Necesară",
            "Te rog introdu cheia API Google Gemini.\nAceasta va fi salvată local într-un fișier .env pentru a nu mai fi cerută.",
            QLineEdit.EchoMode.Password
        )
        
        # Verificăm dacă utilizatorul a apăsat OK și a introdus ceva
        if ok and api_key.strip():
            api_key = api_key.strip()
            try:
                # Creăm și scriem în fișierul .env din folderul rădăcină
                with open(".env", "w", encoding="utf-8") as f:
                    f.write(f'GOOGLE_API_KEY="{api_key}"\n')
                log_timestamp("✅ [API KEY] Cheia a fost salvată cu succes în fișierul .env.", "config")
            except IOError as e:
                # Informăm utilizatorul dacă a apărut o eroare la scrierea fișierului
                log_timestamp(f"❌ [API KEY] Eroare la salvarea fișierului .env: {e}", "config")
                QMessageBox.warning(self, "Eroare Salvare", f"Nu am putut salva cheia API în fișierul .env.\nEroare: {e}\nVa trebui să o introduci din nou data viitoare.")
        
        # Returnăm cheia și statusul pentru a fi folosite în sesiunea curentă
        return api_key, ok    

    def init_ui(self):
        self.setWindowTitle("🎤 Chat Vocal Avansat cu Gemini AI")
        self.setMinimumSize(900, 700)
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        conversation_tab = self.create_conversation_tab()
        audio_tab = self.create_audio_tab()
        ai_settings_tab = self.create_ai_settings_tab()
        self.tabs.addTab(conversation_tab, "💬 Conversație")
        self.tabs.addTab(audio_tab, "🎤 Audio")
        self.tabs.addTab(ai_settings_tab, "🤖 Setări AI")
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def create_conversation_tab(self):
        """Creează tab-ul principal de conversație"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # --- SECȚIUNEA 1: SEMAFOR, STATUS ȘI ASISTENT DESKTOP ---
        status_layout = QHBoxLayout()
        
        # Grup Semafor (stânga)
        semafor_group = QGroupBox("🚦 Semafor")
        semafor_layout = QHBoxLayout()
        
        # Semafor Roșu
        rosu_container = QWidget()
        rosu_container_layout = QVBoxLayout(rosu_container)
        rosu_container_layout.setContentsMargins(0, 0, 0, 0)
        self.semafor_rosu = QLabel()
        self.semafor_rosu.setFixedSize(40, 40)
        self.semafor_rosu.setStyleSheet("background-color: #FF0000; border-radius: 20px;")
        rosu_container_layout.addWidget(self.semafor_rosu)
        
        # Semafor Galben (cu cronometru)
        galben_container = QWidget()
        galben_container_layout = QVBoxLayout(galben_container)
        galben_container_layout.setContentsMargins(0, 0, 0, 0)
        galben_container_layout.setSpacing(2)
        self.semafor_galben = QLabel()
        self.semafor_galben.setFixedSize(40, 40)
        self.semafor_galben.setStyleSheet("background-color: #4A3A00; border-radius: 20px;")
        self.cronometru_galben = QLabel("0.0")
        self.cronometru_galben.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cronometru_galben.setStyleSheet("color: #FFA500; font-size: 10px; font-weight: bold;")
        self.cronometru_galben.hide()
        galben_container_layout.addWidget(self.semafor_galben)
        galben_container_layout.addWidget(self.cronometru_galben)
        
        # Semafor Verde (cu cronometru)
        verde_container = QWidget()
        verde_container_layout = QVBoxLayout(verde_container)
        verde_container_layout.setContentsMargins(0, 0, 0, 0)
        verde_container_layout.setSpacing(2)
        self.semafor_verde = QLabel()
        self.semafor_verde.setFixedSize(40, 40)
        self.semafor_verde.setStyleSheet("background-color: #004A00; border-radius: 20px;")
        self.cronometru_verde = QLabel("15")
        self.cronometru_verde.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cronometru_verde.setStyleSheet("color: #00FF00; font-size: 10px; font-weight: bold;")
        self.cronometru_verde.hide()
        verde_container_layout.addWidget(self.semafor_verde)
        verde_container_layout.addWidget(self.cronometru_verde)
        
        semafor_layout.addWidget(rosu_container)
        semafor_layout.addWidget(galben_container)
        semafor_layout.addWidget(verde_container)
        semafor_group.setLayout(semafor_layout)
        
        # Grup Status (mijloc)
        status_group = QGroupBox("📊 Status")
        status_inner_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("Gata de pornire")
        self.status_label.setStyleSheet("color: #95a5a6; font-size: 14px; font-weight: bold;")
        status_inner_layout.addWidget(self.status_label)
        
        # Grup Asistent Desktop (dreapta) - NOU!
        assistant_group = QGroupBox("🤖 Desktop AI")
        assistant_inner_layout = QVBoxLayout(assistant_group)
        assistant_inner_layout.setContentsMargins(5, 5, 5, 5)
        
        self.desktop_assistant_checkbox = QCheckBox("Activează")
        self.desktop_assistant_checkbox.setToolTip("Când este activat, la fiecare mesaj va fi atașat un screenshot al ecranului principal.")
        self.desktop_assistant_checkbox.setStyleSheet("font-size: 11px; font-weight: bold;")
        self.desktop_assistant_checkbox.setChecked(self.desktop_assistant_mode)
        self.desktop_assistant_checkbox.stateChanged.connect(self.on_desktop_assistant_toggled)
        assistant_inner_layout.addWidget(self.desktop_assistant_checkbox)
        
        # Adăugăm cele 3 grupuri în status_layout
        status_layout.addWidget(semafor_group)
        status_layout.addWidget(status_group, 1)  # stretch factor 1 - se întinde
        status_layout.addWidget(assistant_group)  # dimensiune fixă
        
        # --- SECȚIUNEA 2: BUTOANE CONTROL ---
        buttons_layout = QHBoxLayout()
        
        self.voice_toggle_button = QPushButton("🟢 Activează Microfonul")
        self.voice_toggle_button.setStyleSheet("background-color: #27ae60; font-size: 14px; padding: 10px; font-weight: bold;")
        self.voice_toggle_button.clicked.connect(self.toggle_voice)
        
        self.stop_button = QPushButton("⏹️ Stop Redare")
        self.stop_button.setStyleSheet("background-color: #c0392b; color: white; font-size: 14px; padding: 10px; font-weight: bold;")
        self.stop_button.clicked.connect(self.stop_audio_playback)
        self.stop_button.setEnabled(False)
        
        self.mute_button = QPushButton("🔇 Mute")
        self.mute_button.setStyleSheet("background-color: #f39c12; font-size: 14px; padding: 10px; font-weight: bold;")
        self.mute_button.clicked.connect(self.toggle_mute)
        self.mute_button.setEnabled(False)
        
        buttons_layout.addWidget(self.voice_toggle_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.mute_button)
        
        # --- SECȚIUNEA 3: AFIȘAJ CHAT ---
        chat_group = QGroupBox("💬 Conversație")
        chat_layout = QVBoxLayout(chat_group)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)


        # Adăugăm CSS pentru a formata blocurile de cod și a îmbunătăți aspectul general
        self.chat_display.document().setDefaultStyleSheet("""
            p { margin: 0; padding: 2px; }
            pre {
                background-color: #1e1e1e; /* Culoare de fundal similară cu IDE-urile */
                color: #d4d4d4;           /* Culoare text deschisă */
                padding: 10px;
                border-radius: 5px;
                font-family: 'Courier New', Courier, monospace;
                white-space: pre-wrap;     /* Asigură împachetarea textului */
                display: block;
            }
            code {
                font-family: 'Courier New', Courier, monospace;
            }
        """)

        self.chat_display.setStyleSheet("background-color: #2c3e50; color: white; font-size: 12px; padding: 10px;")
        chat_layout.addWidget(self.chat_display)
        
        # --- SECȚIUNEA 4: INPUT TEXT ---
        input_layout = QHBoxLayout()
        
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Scrie un mesaj sau folosește microfonul...")
        self.text_input.setStyleSheet("font-size: 13px; padding: 8px;")
        self.text_input.returnPressed.connect(self.send_text_message)
        
        self.send_button = QPushButton("📤 Trimite")
        self.send_button.setStyleSheet("background-color: #3498db; color: white; font-size: 13px; padding: 8px 15px; font-weight: bold;")
        self.send_button.clicked.connect(self.send_text_message)
        
        input_layout.addWidget(self.text_input)
        input_layout.addWidget(self.send_button)
        
        # --- ASAMBLARE FINALĂ LAYOUT ---
        layout.addLayout(status_layout)
        layout.addLayout(buttons_layout)
        layout.addWidget(chat_group, 1)
        layout.addLayout(input_layout)
        
        return widget

    @Slot(int)
    @Slot(int)
    def on_desktop_assistant_toggled(self, state):
        """Activează sau dezactivează modul Asistent Desktop."""
        # DEBUG: Vedem ce primim exact
        log_timestamp(f"🔍 [DEBUG] Checkbox state primit: {state} (tip: {type(state)})", "app")
        log_timestamp(f"🔍 [DEBUG] Qt.CheckState.Checked = {Qt.CheckState.Checked} ({Qt.CheckState.Checked.value})", "app")
        log_timestamp(f"🔍 [DEBUG] Qt.CheckState.Unchecked = {Qt.CheckState.Unchecked} ({Qt.CheckState.Unchecked.value})", "app")
        
        # Verificăm dacă state este 2 (Checked) - cea mai sigură metodă
        self.desktop_assistant_mode = (state == Qt.CheckState.Checked.value) or (state == 2)
        
        mode_text = "activat" if self.desktop_assistant_mode else "dezactivat"
        log_timestamp(f"🤖 [ASSISTANT] Modul Asistent Desktop {mode_text}.", "app")
        log_timestamp(f"🔍 [DEBUG] desktop_assistant_mode setat la: {self.desktop_assistant_mode}", "app")
        self.save_config()  # <-- SALVĂM CONFIGURAȚIA


    @Slot()
    def stop_audio_playback(self):
        """Oprește forțat redarea audio și resetează complet starea vocală."""
        log_timestamp("⏹️ [APP] Redarea audio a fost oprită manual de utilizator. Se resetează starea.", "app")
        
        # 1. Oprește sunetul
        self.streaming_tts.stop_all()
        
        # 2. Oprește complet modul vocal, dacă este activ
        if self.voice_enabled:
            # Apelăm funcția principală de comutare pentru a executa oprirea completă
            # și a reseta corect interfața grafică.
            self.toggle_voice()
        
        # 3. O măsură de siguranță pentru a dezactiva butonul de stop
        self.stop_button.setEnabled(False)

    def create_ai_settings_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        prompt_group = QGroupBox("📝 Personalitate AI")
        prompt_layout = QVBoxLayout()
        
        info_label = QLabel(f"Definește personalitatea AI-ului prin intermediul unui system prompt.\nFișier extern: <b>{self.PROMPT_FILE}</b> (poate fi editat direct)")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 11px; color: #666; margin-bottom: 5px;")
        prompt_layout.addWidget(info_label)
        
        # Layout pentru butoane (orizontal)
        buttons_layout = QHBoxLayout()
        
        self.edit_prompt_button = QPushButton("✏️ Editează Prompt")
        self.edit_prompt_button.setStyleSheet("background-color: #3498db; color: white; font-size: 13px; padding: 8px; font-weight: bold;")
        self.edit_prompt_button.clicked.connect(self.open_prompt_editor)
        buttons_layout.addWidget(self.edit_prompt_button)
        
        # BUTON NOU: Reîncarcă din fișier
        self.reload_prompt_button = QPushButton("🔄 Reîncarcă din Fișier")
        self.reload_prompt_button.setStyleSheet("background-color: #27ae60; color: white; font-size: 13px; padding: 8px; font-weight: bold;")
        self.reload_prompt_button.setToolTip(f"Reîncarcă prompt-ul din {self.PROMPT_FILE}\n(util dacă ai editat fișierul direct)")
        self.reload_prompt_button.clicked.connect(self.reload_system_prompt)
        buttons_layout.addWidget(self.reload_prompt_button)
        
        prompt_layout.addLayout(buttons_layout)
        
        self.prompt_preview = QLabel("Prompt actual: (se încarcă...)")
        self.prompt_preview.setWordWrap(True)
        self.prompt_preview.setStyleSheet("font-size: 10px; color: #888; font-style: italic; padding: 5px; background-color: #f5f5f5; border-radius: 3px;")
        prompt_layout.addWidget(self.prompt_preview)
        
        prompt_group.setLayout(prompt_layout)
        
        # --- GRUP NOU: MODEL AI ---
        model_group = QGroupBox("🤖 Model AI")
        model_layout = QFormLayout()
        model_info = QLabel("Selectează modelul AI folosit pentru conversație:")
        model_info.setWordWrap(True)
        model_info.setStyleSheet("font-size: 11px; color: #666; margin-bottom: 5px;")
        model_layout.addRow(model_info)
        
        self.model_combo = QComboBox()
        for model_name in self.available_models.keys():
            self.model_combo.addItem(model_name)
        
        # Setăm modelul curent din config
        for idx, (name, code) in enumerate(self.available_models.items()):
            if code == self.selected_model:
                self.model_combo.setCurrentIndex(idx)
                break
        
        self.model_combo.setStyleSheet("font-size: 12px; padding: 5px;")
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addRow("Model AI:", self.model_combo)
        
        # Descrieri modele
        model_desc = QLabel(
            "<b>Gemini Flash:</b> Rapid și eficient, ideal pentru conversații zilnice<br>"
            "<b>Gemini Pro:</b> Mai avansat, răspunsuri mai detaliate și complexe"
        )
        model_desc.setWordWrap(True)
        model_desc.setStyleSheet("font-size: 10px; color: #777; font-style: italic; margin-top: 5px;")
        model_layout.addRow(model_desc)
        
        model_group.setLayout(model_layout)
        # --- SFÂRȘIT GRUP MODEL AI ---
        
        memory_group = QGroupBox("🧠 Memorie Conversație")
        memory_layout = QFormLayout()
        memory_info = QLabel("Numărul de schimburi de replici pe care AI-ul le păstrează în memorie:")
        memory_info.setWordWrap(True)
        memory_info.setStyleSheet("font-size: 11px; color: #666; margin-bottom: 5px;")
        memory_layout.addRow(memory_info)
        self.memory_spinbox = QSpinBox()
        self.memory_spinbox.setRange(1, 50)
        self.memory_spinbox.setValue(10)
        self.memory_spinbox.setSuffix(" replici")
        self.memory_spinbox.setStyleSheet("font-size: 12px; padding: 5px;")
        self.memory_spinbox.valueChanged.connect(self.on_memory_changed)
        memory_layout.addRow("Limită memorie:", self.memory_spinbox)
        memory_group.setLayout(memory_layout)
        layout.addWidget(prompt_group)
        layout.addWidget(model_group)  # <-- ADĂUGĂM GRUPA MODEL AI
        layout.addWidget(memory_group)
        layout.addStretch()
        widget.setLayout(layout)
        return widget


    def on_threshold_changed(self, value):
        self.voice_config["threshold"] = value
        self.threshold_label.setText(str(value))
        self.save_config()
    def on_pause_changed(self, value):
        self.voice_config["pause_duration"] = value / 10.0
        self.pause_label.setText(f"{value/10.0:.1f}s")
        if self.voice_worker:
            self.voice_worker.pause_duration = value / 10.0
            self.voice_worker.silence_frames_threshold = int((value / 10.0 * 1000) / self.voice_worker.frame_duration)
        self.save_config()
    def on_max_speech_changed(self, value):
        self.voice_config["max_speech_duration"] = value
        self.max_speech_label.setText(f"{value}s")
        if self.voice_worker:
            self.voice_worker.set_max_speech_duration(value)
        self.save_config()
    def on_echo_changed(self, state):
        self.voice_config["enable_echo_cancellation"] = (state == Qt.CheckState.Checked)
        if self.voice_worker:
            self.voice_worker.enable_echo_cancellation = self.voice_config["enable_echo_cancellation"]
        self.save_config()
    def toggle_mute(self):
        if not self.voice_worker: return
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.mute_button.setText("🟢 Activează")
            self.mute_button.setStyleSheet("background-color: #27ae60; font-size: 14px; padding: 10px; font-weight: bold;")
            self.voice_worker.set_muted(True, is_ai_speaking=False)
        else:
            self.mute_button.setText("🔇 Mute")
            self.mute_button.setStyleSheet("background-color: #f39c12; font-size: 14px; padding: 10px; font-weight: bold;")
            self.voice_worker.set_muted(False, is_ai_speaking=False)
    def on_tts_changed(self, state):
        self.tts_enabled = (state == Qt.CheckState.Checked)
        self.save_config()
    def on_voice_changed(self, voice_name):
        self.selected_voice = self.romanian_voices[voice_name]
        self.save_config()
    def open_prompt_editor(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("✏️ Editează Prompt-ul de Sistem")
        dialog.setMinimumSize(600, 400)
        layout = QVBoxLayout()
        
        info_label = QLabel(f"Definește personalitatea și comportamentul AI-ului.\nFișier: {self.PROMPT_FILE}")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        prompt_editor = QTextEdit()
        prompt_editor.setPlainText(self.custom_system_prompt)
        layout.addWidget(prompt_editor)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_prompt = prompt_editor.toPlainText().strip()
            if new_prompt:
                self.custom_system_prompt = new_prompt
                
                # SALVĂM ÎN FIȘIER EXTERN (nu mai salvăm în config)
                if self.save_system_prompt():
                    # Reinițializăm modelul (folosim modelul selectat)
                    self.model = genai.GenerativeModel(model_name=self.selected_model, system_instruction=self.custom_system_prompt)
                    self.chat = self.model.start_chat(history=[])
                    self.conversation_history = []
                    
                    # Actualizăm preview-ul
                    preview_text = new_prompt[:100] + "..." if len(new_prompt) > 100 else new_prompt
                    self.prompt_preview.setText(f"Prompt actual: {preview_text}")
                    
                    QMessageBox.information(self, "Succes", f"Prompt-ul a fost salvat în {self.PROMPT_FILE}!\nConversația a fost resetată.\n\nPoți edita fișierul direct cu orice editor de text.")
                else:
                    QMessageBox.warning(self, "Eroare", f"Nu s-a putut salva prompt-ul în {self.PROMPT_FILE}")
    def on_memory_changed(self, value):
        self.conversation_memory_limit = value
        self.save_config()
    
    def on_model_changed(self, model_name):
        """Handler pentru schimbarea modelului AI."""
        new_model = self.available_models[model_name]
        if new_model != self.selected_model:
            self.selected_model = new_model
            log_timestamp(f"🤖 [MODEL] Model schimbat la: {self.selected_model}", "config")
            
            # Reinițializăm modelul cu noul model selectat
            self.model = genai.GenerativeModel(model_name=self.selected_model, system_instruction=self.custom_system_prompt)
            self.chat = self.model.start_chat(history=[])
            self.conversation_history = []
            
            self.save_config()
            log_timestamp(f"✅ [MODEL] Model reinițializat. Conversația a fost resetată.", "config")
            QMessageBox.information(self, "Model Schimbat", f"Modelul AI a fost schimbat la {model_name}.\nConversația a fost resetată.")
    
    def send_text_message(self):
        text = self.text_input.text().strip()
        if not text: return
        self.add_to_chat("Tu", text)
        self.text_input.clear()
        threading.Thread(target=self.get_gemini_response, args=(text,), daemon=True).start()
    @Slot(str)
    def on_transcription_ready(self, text):
        self.add_to_chat("Tu", text)
        threading.Thread(target=self.get_gemini_response, args=(text,), daemon=True).start()
    @Slot(str)
    def display_gemini_response(self, response_text):
        self.add_to_chat("Gemini", response_text)
    @Slot(bool)
    def on_speech_activity_changed(self, is_speaking):
        if not self.voice_enabled: return
        if is_speaking: self._update_semafor("verde")
        else: self._update_semafor("galben")
    @Slot(int)
    def on_pause_progress_updated(self, progress):
        if not self.voice_enabled or not self.voice_worker: return
        if progress < 100 and self.voice_worker.is_speech_active:
            self._update_semafor("galben")
            timp_ramas = self.voice_config['pause_duration'] * progress / 100.0
            self.cronometru_galben.setText(f"{timp_ramas:.1f}")
            self.cronometru_galben.show()
        elif self.voice_worker.is_speech_active:
            self._update_semafor("verde")
    @Slot(float)
    def on_speech_time_updated(self, timp_ramas):
        if not self.voice_enabled: return
        if timp_ramas >= 0:
            self.cronometru_verde.setText(str(int(timp_ramas)))
            self.cronometru_verde.show()
        else:
            self.cronometru_verde.hide()
    @Slot()
    def on_speech_timeout(self):
        log_timestamp("⏰ [TIMEOUT] Limită timp atinsă", "app")
    @Slot(str)
    def update_status(self, text):
        self.status_label.setText(text)
        if not self.voice_enabled:
            self._update_semafor("rosu")
            return
        if "Aștept să vorbești" in text or "Vorbești" in text:
            self._update_semafor("verde")
        elif any(s in text for s in ["Pauză", "Pausat", "Transcriu", "Aștept răspunsul"]):
            self._update_semafor("rosu")
    def _update_semafor(self, stare):
        self.semafor_rosu.setStyleSheet("background-color: #4A0000; border-radius: 20px;")
        self.semafor_verde.setStyleSheet("background-color: #004A00; border-radius: 20px;")
        self.semafor_galben.setStyleSheet("background-color: #4A3A00; border-radius: 20px;")
        if stare == "rosu":
            self.semafor_rosu.setStyleSheet("background-color: #FF0000; border-radius: 20px;")
            self.cronometru_verde.hide()
            self.cronometru_galben.hide()
        elif stare == "verde":
            self.semafor_verde.setStyleSheet("background-color: #00FF00; border-radius: 20px;")
            self.cronometru_verde.show()
            self.cronometru_galben.hide()
        elif stare == "galben":
            self.semafor_galben.setStyleSheet("background-color: #FFA500; border-radius: 20px;")
            self.cronometru_galben.show()
            self.cronometru_verde.hide()


    def add_to_chat(self, user, message):
        """Adaugă mesaj în chat cu formatare Markdown, culori și auto-scroll."""
        
        # Mută cursorul la sfârșitul documentului pentru a adăuga conținut nou
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)

        # Determinăm culoarea și numele afișat în funcție de utilizator
        if user == "Tu":
            color = "#2980b9"
            display_name = user
        elif user == "Gemini":
            color = "#8e44ad"
            model_display_name = "Flash" if "flash" in self.selected_model.lower() else "Pro"
            display_name = f"Gemini {model_display_name}"
        else:
            color = "#16a085"
            display_name = user

        # Creăm antetul mesajului (ex: "Tu:", "Gemini Flash:")
        header_html = f"<b style='color:{color};'>{display_name}:</b>"

        # Convertim mesajul din Markdown în HTML.
        # 'fenced_code' - pentru blocuri de cod (```)
        # 'nl2br' - convertește newline-urile (\n) în tag-uri <br> pentru a păstra paragrafele
        message_html = markdown.markdown(message, extensions=['fenced_code', 'nl2br'])

        # Inserăm antetul și mesajul formatat
        # Folosim insertHtml pentru a păstra formatarea
        self.chat_display.insertHtml(f"{header_html}<br>{message_html}<br>")

        # Asigurăm auto-scroll la ultimul mesaj
        self.chat_display.ensureCursorVisible()



    def closeEvent(self, event):
        log_timestamp("🛑 Se închide aplicația...", "app")
        self.save_config()
        self.streaming_tts.stop_all()
        if self.voice_worker: self.voice_worker.stop()
        if self.voice_thread:
            self.voice_thread.quit()
            self.voice_thread.wait()
        pygame.mixer.quit()
        event.accept()

if __name__ == "__main__":
    log_timestamp("=" * 60, "app")
    log_timestamp("🎤 CHAT VOCAL AVANSAT CU GEMINI AI (STREAMING) 🎤", "app")
    log_timestamp("=" * 60, "app")
    
    cleanup_temp_files() # <-- AICI ESTE LINIA NOUĂ
    
    app = QApplication(sys.argv)
    window = AdvancedVoiceChatApp()
    window.show()
    sys.exit(app.exec())