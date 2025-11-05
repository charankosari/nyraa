import os
import json
import uuid
import logging
import asyncio
import base64
from dotenv import load_dotenv

load_dotenv()  # Load .env variables

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sarvamai import AsyncSarvamAI, AudioOutput
from openai import AsyncOpenAI
from datetime import datetime

# --- Configuration ---
router = APIRouter()
DATA_DIR = "data"
TEMP_DIR = "temp"  # Make sure this folder exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Sarvam voice map
VOICE_MAP = {
    "hi-IN": "anushka",
    "en-IN": "anushka",
    "te-IN": "anushka",
    "ta-IN": "anushka",
}
DEFAULT_VOICE = "anushka"

# Initialize Async Clients
try:
    sarvam_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize AI clients: {e}")

# Default logging level: INFO. Switch to DEBUG while troubleshooting to see chunk logs.
logging.basicConfig(level=logging.INFO)


async def run_batch_stt_pipeline(audio_file_path: str, request_id: str, websocket: WebSocket):
    """
    Runs the full pipeline using Batch STT (for auto-detect)
    and Streaming TTS (for response).
    """
    try:
        # === 1. Speech-to-Text (Batch API) ===
        logging.info(f"[{request_id}] Starting BATCH STT with file: {audio_file_path}...")

        user_transcript = ""
        language_code = "te-IN"  # Default, will be overwritten

        try:
            with open(audio_file_path, "rb") as f:
                # Use the Batch STT API (transcribe)
                # We do NOT pass language_code, to force auto-detection.
                stt_response = await sarvam_client.speech_to_text.transcribe(
                    file=f,
                    model="saarika:v2.5",
                )

            logging.info(f"[{request_id}] RAW BATCH STT RESPONSE: {stt_response}")

            if getattr(stt_response, "transcript", None):
                user_transcript = stt_response.transcript
                language_code = getattr(stt_response, "language_code", language_code) or language_code
            else:
                logging.warning(f"[{request_id}] STT returned no transcript.")
                await websocket.send_json({"type": "error", "message": "I heard you, but couldn't understand."})
                return

        except Exception as e:
            logging.error(f"[{request_id}] Error during STT API call: {e}")
            await websocket.send_json({"type": "error", "message": "Failed to transcribe audio."})
            return

        # Send the final transcript to the client
        await websocket.send_json({"type": "final_transcript", "text": user_transcript})

        # --- The rest of the pipeline ---
        logging.info(f"[{request_id}] Processing with detected language: [{language_code}]")
        logging.info(f"[{request_id}] User said: [{user_transcript}]")

        # === 2. LLM (OpenAI) ===
        logging.info(f"[{request_id}] Sending to LLM...")
        system_prompt = f"You are a helpful assistant. You MUST respond ONLY in the language identified by the code '{language_code}'."

        llm_stream = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_transcript},
            ],
            stream=True,
        )

        full_llm_response = ""

        # === 3. TTS (Sarvam Streaming) ===
        logging.info(f"[{request_id}] Connecting to Streaming TTS...")
        voice_id = VOICE_MAP.get(language_code, DEFAULT_VOICE)
        logging.info(f"[{request_id}] Using voice: {voice_id} for language: {language_code}")

        async with sarvam_client.text_to_speech_streaming.connect(model="bulbul:v2") as tts_ws:
            await tts_ws.configure(
                target_language_code=language_code,
                speaker=voice_id,
                output_audio_codec="mp3_44100_128",
            )

            async def tts_audio_receiver():
                try:
                    async for tts_response in tts_ws:
                        if isinstance(tts_response, AudioOutput):
                            audio_chunk = base64.b64decode(tts_response.data.audio)
                            # Send raw audio bytes to client
                            await websocket.send_bytes(audio_chunk)
                except Exception as e:
                    logging.info(f"[{request_id}] TTS receiver closed / error: {e}")

            tts_receiver_task = asyncio.create_task(tts_audio_receiver())

            try:
                # Iterate LLM stream safely and robustly
                try:
                    async for chunk in llm_stream:
                        # defensive extraction - works whether chunk.choices[0].delta is dict-like or object-like
                        try:
                            choice0 = chunk.choices[0]
                        except Exception:
                            choice0 = None

                        delta = None
                        if choice0 is not None:
                            # object-like with attributes
                            delta = getattr(choice0, "delta", None)
                            if delta is None and isinstance(choice0, dict):
                                delta = choice0.get("delta")

                        text_chunk = None
                        if delta is not None:
                            if isinstance(delta, dict):
                                # common dict shapes: {"content": "..."} or {"role": "..."} etc.
                                text_chunk = delta.get("content")
                            else:
                                # object-like
                                text_chunk = getattr(delta, "content", None)

                        if text_chunk:
                            # accumulate and forward to TTS
                            full_llm_response += text_chunk
                            logging.debug(f"[{request_id}] LLM chunk: {text_chunk!r}")
                            # convert to speech (may produce TTS audio)
                            try:
                                await tts_ws.convert(text_chunk)
                            except Exception as e:
                                logging.exception(f"[{request_id}] Error calling tts_ws.convert(): {e}")

                except Exception as e:
                    # Log any error that happens while iterating the LLM stream
                    logging.exception(f"[{request_id}] Error while reading LLM stream: {e}")

                # ensure remaining TTS audio is flushed
                try:
                    await tts_ws.flush()
                except Exception as e:
                    logging.exception(f"[{request_id}] Error flushing TTS stream: {e}")

            finally:
                # wait briefly to allow tts_receiver to receive last audio frames
                await asyncio.sleep(1)
                tts_receiver_task.cancel()
                try:
                    await tts_receiver_task
                except asyncio.CancelledError:
                    pass

        # Log the full LLM response (always once)
        final_text = full_llm_response.strip()
        if final_text:
            logging.info(f"[{request_id}] Full LLM Response received: {final_text}")
        else:
            logging.warning(f"[{request_id}] LLM response was empty. No audio was generated.")

        # === 4. Save History ===
        history_data = {
            "requestId": request_id,
            "languageCode": language_code,
            "userPrompt": user_transcript,
            "llmResponse": full_llm_response.strip(),
            "createdAt": datetime.utcnow().isoformat(),
        }
        history_filepath = os.path.join(DATA_DIR, f"{request_id}.json")
        with open(history_filepath, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=4)

        logging.info(f"[{request_id}] Pipeline complete.")

    except Exception as e:
        error_str = str(e)
        # These are "normal" websocket close messages, not errors.
        if "1000 (OK)" in error_str or "1001 (going away)" in error_str:
            logging.info(f"[{request_id}] TTS stream finished and closed normally.")
        else:
            # This is a real, unexpected error
            logging.error(f"[{request_id}] Error in STT/LLM/TTS pipeline: {e}")
            try:
                await websocket.send_json({"type": "error", "message": "Pipeline error."})
            except:
                pass


@router.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket):
    """
    Main WebSocket endpoint.
    Receives audio chunks, saves them, and calls the Batch STT pipeline.
    """
    await websocket.accept()
    request_id = str(uuid.uuid4())
    logging.info(f"[{request_id}] WebSocket connection established.")

    audio_buffer = []
    temp_files = []

    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                logging.info(f"[{request_id}] Client disconnected (inside loop).")
                break

            if "bytes" in message:
                audio_buffer.append(message["bytes"])

            elif "text" in message:
                data = json.loads(message["text"])

                if data.get("type") == "stop_speaking":
                    if not audio_buffer:
                        logging.warning(f"[{request_id}] 'stop_speaking' received with no audio.")
                        continue

                    logging.info(f"[{request_id}] Received stop signal. Processing {len(audio_buffer)} chunks.")

                    # 1. Save buffered audio to a .webm file
                    # The Batch STT API can handle webm directly. No ffmpeg!
                    input_audio_path = os.path.join(TEMP_DIR, f"{request_id}_input.webm")
                    temp_files.append(input_audio_path)

                    with open(input_audio_path, "wb") as f:
                        for chunk in audio_buffer:
                            f.write(chunk)

                    # --- NEW LOGGING ---
                    # Log the file size for easier debugging
                    try:
                        file_size = os.path.getsize(input_audio_path)
                        logging.info(f"[{request_id}] Saved audio to {input_audio_path}. Size: {file_size} bytes.")
                        # Warn if the file is suspiciously small (e.g., < 1KB)
                        if file_size < 1024:
                            logging.warning(
                                f"[{request_id}] Audio file is very small ({file_size} bytes). May result in empty transcript."
                            )
                    except OSError as e:
                        logging.error(f"[{request_id}] Could not get file size: {e}")

                    audio_buffer = []

                    # 2. Run the full AI pipeline
                    logging.info(f"[{request_id}] Starting pipeline task... (will wait for it)")
                    await run_batch_stt_pipeline(input_audio_path, request_id, websocket)
                    logging.info(f"[{request_id}] Pipeline task finished. Ready for next audio.")

    except WebSocketDisconnect:
        logging.info(f"[{request_id}] Client disconnected (outside loop).")
    except Exception as e:
        error_str = str(e)
        # Catch common "normal" disconnect errors (from client/browser) and log as info
        if "Cannot call 'receive' once a disconnect message has been received" in error_str or "1000 (OK)" in error_str or "1001 (going away)" in error_str:
            logging.info(f"[{request_id}] WebSocket closed normally: {e}")
        else:
            # Log real, unexpected errors
            logging.error(f"[{request_id}] Main WebSocket error: {e}")
    finally:
        logging.info(f"[{request_id}] Cleaning up {len(temp_files)} temp files.")
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
