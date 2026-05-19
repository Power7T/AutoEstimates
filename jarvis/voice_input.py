"""Optional voice input — converts microphone speech to text."""

import threading


def listen_once(timeout: int = 10) -> str | None:
    """
    Record from the microphone and return transcribed text.
    Returns None if speech_recognition or pyaudio is not installed,
    or if no speech was detected.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        return None

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=30)
        return recognizer.recognize_google(audio)
    except Exception:
        return None


class VoiceInputThread:
    """Background thread that listens for voice and puts results in a queue."""

    def __init__(self):
        import queue
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def get_nowait(self) -> str | None:
        import queue
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _run(self):
        while not self._stop.is_set():
            text = listen_once(timeout=5)
            if text:
                self._queue.put(text)
