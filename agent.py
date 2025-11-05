# agent.py
import os
import re
import json
import uuid
import logging
import asyncio
import base64
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse
from openai import AsyncOpenAI

# --- Configuration ---
router = APIRouter()
DATA_DIR = "data"
TEMP_DIR = "temp"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

VOICE_MAP = {
    "hi-IN": "anushka",
    "en-IN": "anushka",
    "te-IN": "anushka",
    "ta-IN": "anushka",
}
DEFAULT_VOICE = "anushka"

HOSPITAL_NAME = os.getenv("HOSPITAL_NAME", "Sunrise Hospital")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "NyraAI")

try:
    sarvam_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize AI clients: {e}")

logging.basicConfig(level=logging.INFO)


def extract_json_from_text(text: str):
    """
    Attempt to extract the first valid JSON object from `text`.
    Returns parsed object or None.
    """
    if not text:
        return None
    # naive but practical approach: search for the first balanced {...}
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    # fallthrough to scanning further starts
                    break
    # fallback: brute force search all pairs (slower)
    starts = [m.start() for m in re.finditer(r"\{", text)]
    ends = [m.start() for m in re.finditer(r"\}", text)]
    for s in starts:
        for e in ends:
            if e <= s:
                continue
            cand = text[s : e + 1]
            try:
                return json.loads(cand)
            except Exception:
                continue
    return None


def remove_json_and_metadata_markers(text: str):
    """
    Remove JSON blocks and <METADATA>...</METADATA> blocks from `text`.
    Returns (cleaned_text, removed_parts_list).
    This ensures we never send the metadata JSON to TTS.
    """
    removed = []

    if not text:
        return text, removed

    # 1) Remove explicit metadata marker blocks if present
    # Case-insensitive tag search
    meta_pattern = re.compile(r"(?is)<metadata>(.*?)</metadata>")
    def _meta_repl(m):
        removed.append(m.group(0))
        return " "  # replace with space to keep spacing
    text = meta_pattern.sub(_meta_repl, text)

    # 2) Remove the first balanced JSON block (if any). We try to find the largest first balanced block near the end
    # (often LLM appends metadata at the end).
    start_idxs = [m.start() for m in re.finditer(r"\{", text)]
    if start_idxs:
        # Try scanning from last opening brace backwards to find a balanced block
        for s in reversed(start_idxs):
            depth = 0
            for i in range(s, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        cand = text[s : i + 1]
                        try:
                            # verify it's valid JSON
                            json.loads(cand)
                            # remove it
                            removed.append(cand)
                            # replace with a single space to keep sentence boundaries
                            text = text[:s] + " " + text[i + 1 :]
                            # only remove first valid JSON block
                            raise StopIteration
                        except StopIteration:
                            break
                        except Exception:
                            break
            else:
                continue
            break

    # Also collapse excessive whitespace
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned, removed


async def run_batch_stt_pipeline(audio_file_path: str, request_id: str, websocket: WebSocket):
    """
    STT -> LLM(stream) -> segmented TTS (short lived per-segment) pipeline.
    Strips metadata JSON from TTS input to avoid speaking it. Sends metadata to client as llm_metadata.
    """
    try:
        logging.info(f"[{request_id}] Starting BATCH STT with file: {audio_file_path}...")

        user_transcript = ""
        language_code = "te-IN"

        try:
            with open(audio_file_path, "rb") as f:
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
            logging.exception(f"[{request_id}] Error during STT call: {e}")
            await websocket.send_json({"type": "error", "message": "Failed to transcribe audio."})
            return

        try:
            await websocket.send_json({"type": "final_transcript", "text": user_transcript})
        except Exception:
            logging.debug(f"[{request_id}] client may be disconnected before final_transcript")

        logging.info(f"[{request_id}] Detected language: {language_code}")
        logging.info(f"[{request_id}] User said: {user_transcript}")

        # LLM system prompt (encourage metadata in explicit marker)
        system_prompt = f"""
You are {ASSISTANT_NAME}, the virtual hospital assistant for {HOSPITAL_NAME}.
Speak only in the detected language. Warm, concise, professional.

IMPORTANT: produce two things:
1) A natural-language reply (for the caller) only.
2) A JSON metadata object for backend use containing intent, fields, emergency_status, confidence, handoff.

**CRITICAL**: append the metadata JSON at the very end wrapped in a marker like:
<METADATA>
{{"intent":"intent_book_appointment","fields":{{...}},"emergency_status":"low","confidence":"high","handoff":false}}
</METADATA>

The server will strip anything inside <METADATA>...</METADATA> before sending text to TTS.
"""

        llm_stream = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_transcript},
            ],
            stream=True,
        )

        full_llm_response = ""
        FLUSH_THRESHOLD_CHARS = 80
        pending_segment = ""

        async def convert_segment_and_forward(segment_text: str):
            # remove JSON/metadata before sending to TTS
            cleaned_text, removed_parts = remove_json_and_metadata_markers(segment_text)
            if removed_parts:
                logging.info(f"[{request_id}] Stripped metadata from segment before TTS: {removed_parts}")

            if not cleaned_text.strip():
                logging.info(f"[{request_id}] After stripping metadata, nothing left to TTS for this segment.")
                return

            audio_bytes_buffer = bytearray()
            try:
                async with sarvam_client.text_to_speech_streaming.connect(
                    model="bulbul:v2", send_completion_event=True
                ) as tts_ws:
                    await tts_ws.configure(
                        target_language_code=language_code,
                        speaker=VOICE_MAP.get(language_code, DEFAULT_VOICE),
                    )

                    try:
                        await tts_ws.convert(cleaned_text)
                        await tts_ws.flush()
                    except Exception as e:
                        logging.exception(f"[{request_id}] convert/flush error: {e}")

                    try:
                        async for msg in tts_ws:
                            if isinstance(msg, AudioOutput):
                                try:
                                    chunk = base64.b64decode(msg.data.audio)
                                    audio_bytes_buffer.extend(chunk)
                                except Exception:
                                    logging.debug(f"[{request_id}] failed decode audiooutput chunk")
                                continue

                            if isinstance(msg, EventResponse):
                                ev = getattr(msg.data, "event_type", None)
                                logging.debug(f"[{request_id}] TTS EventResponse: {ev}")
                                if ev == "final":
                                    break

                            # defensive check
                            data_obj = getattr(msg, "data", None)
                            if data_obj is not None:
                                audio_attr = getattr(data_obj, "audio", None)
                                if audio_attr:
                                    try:
                                        chunk = base64.b64decode(audio_attr)
                                        audio_bytes_buffer.extend(chunk)
                                    except Exception:
                                        pass

                    except Exception as e:
                        logging.info(f"[{request_id}] TTS collection loop ended: {e}")

            except Exception as e:
                logging.exception(f"[{request_id}] Failed to open/use TTS WS: {e}")

            # send blob if available
            try:
                if audio_bytes_buffer:
                    try:
                        await websocket.send_json({"type": "audio_start"})
                    except Exception:
                        logging.debug(f"[{request_id}] couldn't send audio_start")

                    try:
                        await websocket.send_bytes(bytes(audio_bytes_buffer))
                        logging.info(f"[{request_id}] Sent audio blob to client ({len(audio_bytes_buffer)} bytes)")
                    except Exception as e:
                        logging.exception(f"[{request_id}] Failed to send binary blob: {e}")

                    # debug write
                    try:
                        debug_path = os.path.join(os.getcwd(), "debug_last.mp3")
                        with open(debug_path, "wb") as df:
                            df.write(bytes(audio_bytes_buffer))
                        logging.debug(f"[{request_id}] wrote debug file {debug_path}")
                    except Exception:
                        pass

                    try:
                        await websocket.send_json({"type": "audio_end", "bytes": len(audio_bytes_buffer)})
                    except Exception:
                        logging.debug(f"[{request_id}] couldn't send audio_end")
                else:
                    logging.warning(f"[{request_id}] No audio bytes collected for this segment.")
            except Exception as e:
                logging.exception(f"[{request_id}] while sending audio to client: {e}")

        # Stream LLM content
        try:
            async for chunk in llm_stream:
                try:
                    choice0 = chunk.choices[0]
                except Exception:
                    choice0 = None

                delta = None
                if choice0 is not None:
                    delta = getattr(choice0, "delta", None)
                    if delta is None and isinstance(choice0, dict):
                        delta = choice0.get("delta")

                text_chunk = None
                if delta is not None:
                    if isinstance(delta, dict):
                        text_chunk = delta.get("content")
                    else:
                        text_chunk = getattr(delta, "content", None)

                if text_chunk:
                    full_llm_response += text_chunk
                    pending_segment += text_chunk

                    if len(pending_segment) >= FLUSH_THRESHOLD_CHARS or pending_segment.strip().endswith((".", "?", "!", "।")):
                        seg_to_send = pending_segment
                        pending_segment = ""
                        # send partial text to client (optional)
                        try:
                            await websocket.send_json({"type": "llm_response_partial", "text": seg_to_send})
                        except Exception:
                            pass
                        # convert cleaned segment (metadata stripped inside)
                        await convert_segment_and_forward(seg_to_send)

        except Exception as e:
            logging.exception(f"[{request_id}] Error reading LLM stream: {e}")

        # remainder
        if pending_segment.strip():
            try:
                await websocket.send_json({"type": "llm_response_partial", "text": pending_segment})
            except Exception:
                pass
            await convert_segment_and_forward(pending_segment)
            pending_segment = ""

        final_text = full_llm_response.strip()

        # Extract metadata (either from explicit markers or balanced JSON)
        metadata = None
        if final_text:
            # Try marker first
            meta_match = re.search(r"(?is)<metadata>(.*?)</metadata>", final_text)
            if meta_match:
                meta_payload = meta_match.group(1).strip()
                try:
                    metadata = json.loads(meta_payload)
                    logging.info(f"[{request_id}] Parsed metadata via <METADATA> marker")
                except Exception as e:
                    logging.debug(f"[{request_id}] Failed parse metadata marker: {e}; trying generic JSON extraction")
                    metadata = extract_json_from_text(final_text)
            else:
                metadata = extract_json_from_text(final_text)

            # Clean the spoken text (remove metadata before sending final llm_response to UI as text)
            cleaned_final_text, removed_parts = remove_json_and_metadata_markers(final_text)
            if removed_parts:
                logging.info(f"[{request_id}] Removed metadata from final_text before sending/displaying: {removed_parts}")
            # send cleaned final text to client
            try:
                await websocket.send_json({"type": "llm_response", "text": cleaned_final_text})
            except Exception:
                logging.debug(f"[{request_id}] couldn't send final llm_response")

            # send metadata separately if found
            if metadata:
                try:
                    await websocket.send_json({"type": "llm_metadata", "metadata": metadata})
                except Exception:
                    logging.debug(f"[{request_id}] couldn't send llm_metadata")

        else:
            logging.warning(f"[{request_id}] Empty LLM response")

        # Save history (store both raw LLM response and parsed metadata)
        history = {
            "requestId": request_id,
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "languageCode": language_code,
            "userTranscript": user_transcript,
            "llmRaw": full_llm_response,
            "llmCleaned": cleaned_final_text if final_text else "",
            "llmMetadata": metadata,
        }
        try:
            path = os.path.join(DATA_DIR, f"{request_id}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(history, fh, ensure_ascii=False, indent=2)
            logging.info(f"[{request_id}] Saved history to {path}")
        except Exception as e:
            logging.exception(f"[{request_id}] Failed to save history: {e}")

    except Exception as e:
        error_str = str(e)
        if "1000 (OK)" in error_str or "1001 (going away)" in error_str:
            logging.info(f"[{request_id}] normal close: {e}")
        else:
            logging.exception(f"[{request_id}] Pipeline error: {e}")
            try:
                await websocket.send_json({"type": "error", "message": "Pipeline error."})
            except Exception:
                pass


@router.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket):
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
                try:
                    data = json.loads(message["text"])
                except Exception:
                    logging.warning(f"[{request_id}] Received invalid JSON text message.")
                    continue

                if data.get("type") == "stop_speaking":
                    if not audio_buffer:
                        logging.warning(f"[{request_id}] 'stop_speaking' with no audio.")
                        continue

                    logging.info(f"[{request_id}] Received stop signal. Processing {len(audio_buffer)} chunks.")
                    input_audio_path = os.path.join(TEMP_DIR, f"{request_id}_input.webm")
                    temp_files.append(input_audio_path)
                    with open(input_audio_path, "wb") as f:
                        for chunk in audio_buffer:
                            f.write(chunk)

                    try:
                        file_size = os.path.getsize(input_audio_path)
                        logging.info(f"[{request_id}] Saved audio {input_audio_path} size={file_size}")
                    except Exception:
                        logging.exception(f"[{request_id}] stat error on input file")

                    audio_buffer = []

                    await run_batch_stt_pipeline(input_audio_path, request_id, websocket)
                    logging.info(f"[{request_id}] Pipeline finished. Ready for next audio.")

    except WebSocketDisconnect:
        logging.info(f"[{request_id}] Client disconnected (outside loop).")
    except Exception as e:
        error_str = str(e)
        if ("Cannot call 'receive' once a disconnect message has been received" in error_str
                or "1000 (OK)" in error_str or "1001 (going away)" in error_str):
            logging.info(f"[{request_id}] WS closed normally: {e}")
        else:
            logging.exception(f"[{request_id}] Main WS error: {e}")
    finally:
        logging.info(f"[{request_id}] Cleaning up {len(temp_files)} temp files.")
        for p in temp_files:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
