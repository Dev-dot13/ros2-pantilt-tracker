import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import sounddevice as sd
import numpy as np
import webrtcvad
import collections
import threading
import queue
import time
from faster_whisper import WhisperModel


# --- Audio config ---
SAMPLE_RATE    = 16000   # Whisper expects 16kHz
CHANNELS       = 1
FRAME_MS       = 30      # webrtcvad works with 10, 20, or 30ms frames
FRAME_SAMPLES  = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples per frame

# --- Voice activity detection config ---
VAD_AGGRESSIVENESS = 2        # 0-3, higher = more aggressive filtering
SILENCE_THRESHOLD  = 0.8      # seconds of silence before clip is considered done
PRE_SPEECH_BUFFER  = 10       # number of frames to keep before speech starts
MIN_SPEECH_FRAMES  = 5        # minimum frames to consider valid speech

# --- Whisper config ---
WHISPER_MODEL  = 'base'       # tiny/base/small — base is good balance of speed/accuracy
WHISPER_DEVICE = 'cuda'


class VoiceInputNode(Node):

    def __init__(self):
        super().__init__('voice_input_node')

        # Publisher — sends transcribed text to LLaVA
        self.command_pub = self.create_publisher(
            String, '/llm/command', 10)

        # Subscribe to LLaVA responses so we can print them
        self.create_subscription(
            String, '/llm/response',
            self.response_callback, 10)

        # Load Whisper model
        self.get_logger().info(
            f'Loading Whisper {WHISPER_MODEL} on {WHISPER_DEVICE}...')
        self.whisper = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type='float16'   # faster on RTX 4060
        )
        self.get_logger().info('Whisper ready.')

        # VAD
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        # Audio queue — mic thread puts frames here, processing thread reads
        self.audio_queue = queue.Queue()

        # State
        self.is_speaking      = False
        self.speech_frames    = []
        self.silence_frames   = 0
        self.pre_speech_buf   = collections.deque(
            maxlen=PRE_SPEECH_BUFFER)

        # Silence frame count threshold
        self.silence_frame_threshold = int(
            SILENCE_THRESHOLD * 1000 / FRAME_MS)

        # Start mic and processing threads
        self.mic_thread = threading.Thread(
            target=self._mic_stream, daemon=True)
        self.proc_thread = threading.Thread(
            target=self._process_loop, daemon=True)

        self.mic_thread.start()
        self.proc_thread.start()

        self.get_logger().info('Voice input active — speak naturally.')
        print('\n  [Listening... speak a command]\n')

    # ------------------------------------------------------------------
    # LLaVA response — print to terminal
    # ------------------------------------------------------------------

    def response_callback(self, msg: String):
        print(f'\n  Camera: {msg.data}\n  [Listening...]\n')

    # ------------------------------------------------------------------
    # Microphone stream — runs in background thread
    # ------------------------------------------------------------------

    def _mic_stream(self):
        def callback(indata, frames, time_info, status):
            if status:
                self.get_logger().warn(f'Mic status: {status}')
            # Convert to 16-bit PCM bytes — webrtcvad expects this
            pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            self.audio_queue.put(pcm)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            blocksize=FRAME_SAMPLES,
            device=10,            # system default
            callback=callback
        ):
            self.get_logger().info('Microphone stream open.')
            # Keep stream alive until node shuts down
            while rclpy.ok():
                time.sleep(0.1)

    # ------------------------------------------------------------------
    # Processing loop — VAD + Whisper
    # ------------------------------------------------------------------

    def _process_loop(self):
        while rclpy.ok():
            try:
                pcm = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Voice activity detection
            try:
                is_speech = self.vad.is_speech(pcm, SAMPLE_RATE)
            except Exception:
                is_speech = False

            if is_speech:
                if not self.is_speaking:
                    # Speech just started — include pre-speech buffer
                    self.is_speaking   = True
                    self.silence_frames = 0
                    self.speech_frames  = list(self.pre_speech_buf)
                    self.get_logger().debug('Speech started.')
                self.speech_frames.append(pcm)
                self.silence_frames = 0

            else:
                self.pre_speech_buf.append(pcm)

                if self.is_speaking:
                    self.silence_frames += 1

                    if self.silence_frames >= self.silence_frame_threshold:
                        # Silence long enough — clip is complete
                        self.is_speaking = False
                        self.get_logger().debug(
                            f'Speech ended. '
                            f'{len(self.speech_frames)} frames collected.')

                        if len(self.speech_frames) >= MIN_SPEECH_FRAMES:
                            # Transcribe in a separate thread
                            # so we don't block VAD processing
                            frames_copy = list(self.speech_frames)
                            t = threading.Thread(
                                target=self._transcribe,
                                args=(frames_copy,),
                                daemon=True
                            )
                            t.start()

                        self.speech_frames  = []
                        self.silence_frames = 0

    # ------------------------------------------------------------------
    # Whisper transcription
    # ------------------------------------------------------------------

    def _transcribe(self, frames):
        try:
            # Combine PCM frames into float32 numpy array
            pcm_bytes = b''.join(frames)
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(
                np.float32) / 32768.0

            # Minimum duration check — avoid transcribing noise
            duration = len(audio) / SAMPLE_RATE
            if duration < 0.5:
                self.get_logger().debug(
                    f'Clip too short ({duration:.2f}s), skipping.')
                return

            self.get_logger().info(
                f'Transcribing {duration:.1f}s clip...')

            segments, info = self.whisper.transcribe(
                audio,
                language='en',
                beam_size=5,
                vad_filter=True,       # Whisper's own VAD as second filter
                vad_parameters=dict(
                    min_silence_duration_ms=500)
            )

            # Collect all segments into one string
            text = ' '.join(s.text.strip() for s in segments).strip()

            if not text:
                self.get_logger().debug('Whisper returned empty transcript.')
                return

            self.get_logger().info(f'Heard: "{text}"')
            print(f'\n  You said: "{text}"')
            print(f'  [processing...]\n')

            # Publish to /llm/command — LLaVA picks it up
            msg = String()
            msg.data = text
            self.command_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Transcription failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = VoiceInputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()