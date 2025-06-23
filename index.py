import os
import asyncio
import base64
import io
import traceback

import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from google import genai
from google.genai import types

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# Gemini model to use
MODEL = "models/gemini-2.0-flash-live-001"

# Default video input mode
DEFAULT_MODE = "camera"

# Initialize Gemini client
# Ensure GEMINI_API_KEY environment variable is set
client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("AIzaSyBt3W-fHQ-PPQBTPNlvKrC-XdVt_hz97QI"),
)

# Configuration for LiveConnect session
CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        language_code="ar-XA", # Set language to Arabic
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Leda") # Arabic voice
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
)

# Initialize PyAudio
pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self):
        """Allows user to type text messages to send to the AI."""
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ", # Prompt for user input
            )
            if text.lower() == "q": # 'q' to quit
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        """Captures a single frame from the webcam."""
        ret, frame = cap.read()
        if not ret:
            return None
        # Convert BGR to RGB for PIL, preventing color tint
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024]) # Resize image

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        """Continuously captures and sends webcam frames."""
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0) # Send frame every 1 second

            await self.out_queue.put(frame)

        cap.release() # Release the VideoCapture object

    def _get_screen(self):
        """Captures a single screenshot."""
        sct = mss.mss()
        monitor = sct.monitors[0] # Primary monitor

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        """Continuously captures and sends screenshots."""
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0) # Send screenshot every 1 second

            await self.out_queue.put(frame)

    async def send_realtime(self):
        """Sends data (audio/video) from the output queue to the Gemini session."""
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        """Records audio from the microphone and puts it into the output queue."""
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        # Exception handling for audio overflow during debugging
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        """Receives audio and text responses from the Gemini session."""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data) # Put audio data in playback queue
                    continue
                if text := response.text:
                    print(text, end="") # Print text responses

            # Clear audio queue if turn is complete (e.g., due to interruption)
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        """Plays audio chunks from the audio input queue."""
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        """Main function to start and manage all asynchronous tasks."""
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue() # Queue for audio to be played
                self.out_queue = asyncio.Queue(maxsize=5) # Queue for data to be sent

                # Create tasks for different functionalities
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                # Wait for the send_text task to complete (user types 'q')
                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass # User exited gracefully
        except ExceptionGroup as EG:
            # Handle other exceptions and close audio stream if open
            if hasattr(self, 'audio_stream') and self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from (camera, screen, or none)",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())
