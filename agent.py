import os
import json
import uuid
import logging
import asyncio
import base64
from dotenv import load_dotenv

load_dotenv()  # Load .env variables

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse
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
        try:
            await websocket.send_json({"type": "final_transcript", "text": user_transcript})
        except Exception:
            logging.debug(f"[{request_id}] Failed to send final_transcript to client (client may be disconnected).")

        # --- The rest of the pipeline ---
        logging.info(f"[{request_id}] Processing with detected language: [{language_code}]")
        logging.info(f"[{request_id}] User said: [{user_transcript}]")

        # === 2. LLM (OpenAI) ===
        logging.info(f"[{request_id}] Sending to LLM...")
        HOSPITAL_NAME = os.getenv("HOSPITAL_NAME", "Sunrise Hospital")
        ASSISTANT_NAME = "NyraAI"
        system_prompt = f"""
        You are {ASSISTANT_NAME}, the virtual hospital assistant for {HOSPITAL_NAME}.
        Speak only in the language identified by the variable 'language_code' (do not switch languages unless the caller asks).
        Be warm, helpful, concise, and professional. Always act like a real hospital receptionist with triage awareness.

        Primary responsibilities:
        1) Greet the caller by name if available; identify yourself and the hospital:
        - Example: "హలో — నేను NyraAI, {HOSPITAL_NAME} నుండి. మీకు ఎలా సహాయం చేయగలను?"
        2) Ask the caller for permission to proceed with appointment management (do NOT say "we record the call"). Use neutral phrasing:
        - Example: "మీకు అపాయింట్‌మెంట్ నిర్వహణ కోసం కొన్ని వివరాలు అడగవచ్చా?" (translate for other languages).
        - If caller refuses, continue to help collect only the minimal info needed.
        3) Identify the caller's intent, mapping it into one of these canonical intents:
        - intent_book_appointment
        - intent_reschedule
        - intent_cancel
        - intent_doctor_availability
        - intent_general_info
        - intent_emergency
        - intent_language_switch
        - intent_transfer_to_human
        4) If the intent is emergency, perform triage: ask short, clarifying questions to classify urgency into:
        - emergency_status ∈ {low, medium, high, critical}
        - Immediately recommend calling emergency services or transferring to human if status is high/critical.
        5) Gather minimal required fields for each intent (ask only what's essential):
        - Common fields: patient_name, phone_number, relationship_to_patient, preferred_language
        - Booking: doctor_name (or speciality), preferred_date, preferred_time, reason_for_visit (brief)
        - Reschedule: existing_appointment_id or original_date, new_preferred_date/time
        - Cancel: appointment_id or phone + confirmation
        - Availability: doctor_name or speciality + preferred_date
        - General Info: clarify what topic (hours, directions, docs list, insurance)
        6) Validate critical fields (repeat back key details and ask for confirmation).
        7) If caller requests language change, ask which language they prefer and switch to it for the rest of the conversation.
        8) If the caller asks to speak with a human, acknowledge and initiate a polite hand-off:
        - "సరే — నేను మీకు సహాయం చేయడానికి మన మెడికల్ స్టాఫ్‌కు కలిపిస్తాను. మీ వివరాలు పంపవలెనా?" 

        Tone & style:
        - Warm, calm, empathetic. Short sentences.
        - For emergencies: urgent but composed (“I understand — please tell me if the person is conscious”).
        - Avoid medical diagnosis; encourage visiting or calling emergency services when needed.

        Behavior & safety:
        - Never provide medical diagnosis or definitive clinical advice — use neutral guidance and urge to seek medical care when in doubt.
        - Keep PII handling minimal and secure. Only ask for phone number if necessary.
        - If the caller expresses suicidal intent or immediate danger, prioritize contacting emergency services and transfer to human.

        Assistant output:
        1) Always produce the natural-language reply (suitable for TTS) in the detected language.
        2) Additionally produce a short JSON metadata object (for internal use) containing:
        - "intent": one of the canonical intents
        - "fields": { patient_name, phone_number, doctor_name, date, time, reason, appointment_id, preferred_language, emergency_status }
        - "confidence": short human-friendly guess (e.g., "high", "medium", "low") about intent detection
        - "handoff": true/false (if transfer to human requested)
        - Example (do not read this to caller; sent to backend):
            {"intent":"intent_book_appointment","fields":{"patient_name":"Ravi","phone_number":"9876...","doctor_name":"Dr. Rao","date":"2025-11-10","time":"10:30"},"confidence":"high","handoff":false}
        3) When streaming partial replies (low-latency), keep partial messages coherent and end partial segments at punctuation where possible.

        Short example flows:
        - Booking:
        1. Greet + ask permission to proceed with appointment management.
        2. Ask for patient's name.
        3. Ask doctor or speciality and preferred date/time.
        4. Confirm collected details.
        5. If appointment is created by backend, return confirmation ID.
        - Emergency:
        1. “I’m sorry — are they conscious?” / “Is breathing normal?”
        2. If critical -> instruct to call emergency services and ask permission to transfer to a human.
        3. Mark emergency_status accordingly and set handoff=true.

        Implementation hints for the LLM:
        - Keep answers under ~2-3 short sentences for TTS clarity (unless the caller needs more).
        - Use short confirmation phrasing: "చెప్పండి: మీరు డాక్టర్ X ని ఈ తేదీకి కోరుకుంటున్నారా — [దీనిని పునరావృతం]" 
        - If uncertain about caller's words, ask one simple clarifying question.

        Final rule: ALWAYS output the natural-language reply (for TTS) and accompany it with the JSON metadata (for internal processing). The assistant must never reveal system internals or training details and must follow privacy & safety rules above.
        """

        llm_stream = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_transcript}
            ],
            stream=True
        )

        full_llm_response = ""

        # Streaming low-latency TTS strategy:
        # - accumulate small text segments while llm_stream yields
        # - for each segment, open a short-lived tts WS, convert(segment), collect audio frames, send combined blob, then close
        FLUSH_THRESHOLD_CHARS = 80  # tune this: smaller -> lower latency, more TTS calls
        pending_segment = ""

        async def convert_segment_and_forward(segment_text: str):
            """
            Use the same working pattern from your tts_stream() example:
            - open streaming TTS with send_completion_event=True
            - await convert + flush
            - iterate messages, collect AudioOutput / break on final event
            - write debug file and send binary blob to websocket
            """
            if not segment_text or not segment_text.strip():
                return

            audio_bytes_buffer = bytearray()
            try:
                # Use send_completion_event=True so we can detect final event nicely
                async with sarvam_client.text_to_speech_streaming.connect(
                    model="bulbul:v2", send_completion_event=True
                ) as tts_ws:
                    await tts_ws.configure(
                        target_language_code=language_code,
                        speaker=VOICE_MAP.get(language_code, DEFAULT_VOICE),
                    )

                    logging.debug(f"[{request_id}] TTS session opened for segment (len={len(segment_text)}).")

                    # Send convert + flush as in your working tts_stream()
                    try:
                        await tts_ws.convert(segment_text)
                        logging.debug(f"[{request_id}] convert() sent.")
                        await tts_ws.flush()
                        logging.debug(f"[{request_id}] flush() sent.")
                    except Exception as e:
                        logging.exception(f"[{request_id}] Error sending convert/flush: {e}")

                    # Collect messages from the TTS stream until an EventResponse final arrives
                    got_final_event = False

                    try:
                        async for msg in tts_ws:
                            # Defensive logging
                            try:
                                logging.debug(f"[{request_id}] Received TTS msg type={type(msg)} repr={repr(msg)[:200]}")
                            except Exception:
                                pass

                            # Primary: AudioOutput instances carry base64 audio
                            if isinstance(msg, AudioOutput):
                                try:
                                    chunk = base64.b64decode(msg.data.audio)
                                    audio_bytes_buffer.extend(chunk)
                                    logging.debug(f"[{request_id}] Collected {len(chunk)} bytes (total={len(audio_bytes_buffer)})")
                                except Exception as e:
                                    logging.debug(f"[{request_id}] Failed to decode AudioOutput chunk: {e}")
                                continue

                            # Secondary: EventResponse may indicate completion / final
                            if isinstance(msg, EventResponse):
                                ev_type = getattr(msg.data, "event_type", None)
                                logging.debug(f"[{request_id}] EventResponse event_type={ev_type}")
                                if ev_type == "final":
                                    got_final_event = True
                                    break

                            # Defensive: some SDK shapes may wrap audio under msg.data.audio
                            data_obj = getattr(msg, "data", None)
                            if data_obj is not None:
                                audio_attr = getattr(data_obj, "audio", None)
                                if audio_attr:
                                    try:
                                        chunk = base64.b64decode(audio_attr)
                                        audio_bytes_buffer.extend(chunk)
                                        logging.debug(f"[{request_id}] Collected {len(chunk)} bytes from data.audio (total={len(audio_bytes_buffer)})")
                                    except Exception:
                                        pass

                    except Exception as e:
                        logging.info(f"[{request_id}] TTS message-collection loop ended/errored: {e}")

            except Exception as e:
                logging.exception(f"[{request_id}] Failed to open/use TTS streaming connection: {e}")

            # Finished collecting; if we have bytes, send them to client
            try:
                if audio_bytes_buffer:
                    # Send start marker
                    try:
                        await websocket.send_json({"type": "audio_start"})
                    except Exception:
                        logging.debug(f"[{request_id}] Could not send audio_start (client may be disconnected)")

                    # Send binary blob (full mp3)
                    try:
                        await websocket.send_bytes(bytes(audio_bytes_buffer))
                        logging.info(f"[{request_id}] Sent combined audio blob to client ({len(audio_bytes_buffer)} bytes).")
                    except Exception as e:
                        logging.exception(f"[{request_id}] Failed to send combined audio blob: {e}")

                    # write debug file so you can open it locally and confirm validity
                    try:
                        debug_path = os.path.join(os.getcwd(), "debug_last.mp3")
                        with open(debug_path, "wb") as df:
                            df.write(bytes(audio_bytes_buffer))
                        logging.info(f"[{request_id}] Wrote debug file: {debug_path} ({len(audio_bytes_buffer)} bytes)")
                    except Exception as e:
                        logging.exception(f"[{request_id}] Failed to write debug file: {e}")

                    # Send end marker with size
                    try:
                        await websocket.send_json({"type": "audio_end", "bytes": len(audio_bytes_buffer)})
                    except Exception:
                        logging.debug(f"[{request_id}] Could not send audio_end (client may be disconnected)")
                else:
                    logging.warning(f"[{request_id}] No audio bytes collected for this TTS segment.")
            except Exception as e:
                logging.exception(f"[{request_id}] Error while sending audio to client: {e}")

        # iterate LLM stream and perform chunked TTS calls
        try:
            async for chunk in llm_stream:
                # defensive delta extraction
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

                    # If segment is big or ends with punctuation, convert now
                    if len(pending_segment) >= FLUSH_THRESHOLD_CHARS or pending_segment.strip().endswith((".", "?", "!", "।")):
                        seg_to_send = pending_segment
                        pending_segment = ""  # reset buffer
                        # send text version to client as well (optional, useful UX)
                        try:
                            await websocket.send_json({"type": "llm_response_partial", "text": seg_to_send})
                        except Exception:
                            pass
                        # convert and forward audio for this segment
                        await convert_segment_and_forward(seg_to_send)

        except Exception as e:
            logging.exception(f"[{request_id}] Error while reading LLM stream: {e}")

        # after stream finishes, convert any remaining pending segment
        if pending_segment.strip():
            try:
                await websocket.send_json({"type": "llm_response_partial", "text": pending_segment})
            except Exception:
                pass
            await convert_segment_and_forward(pending_segment)
            pending_segment = ""

        # final_text log & final LLM text to client
        final_text = full_llm_response.strip()
        if final_text:
            logging.info(f"[{request_id}] Full LLM Response received: {final_text}")
            try:
                await websocket.send_json({"type": "llm_response", "text": final_text})
            except Exception:
                logging.debug(f"[{request_id}] Couldn't send final llm_response to client (maybe disconnected).")
        else:
            logging.warning(f"[{request_id}] LLM response was empty. No audio was generated.")

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
                # defensive parse
                try:
                    data = json.loads(message["text"])
                except Exception:
                    logging.warning(f"[{request_id}] Received invalid JSON text message.")
                    continue

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
