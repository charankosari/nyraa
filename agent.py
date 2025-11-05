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

# -----------------------
# Config
# -----------------------
router = APIRouter()
DATA_DIR = "data"
TEMP_DIR = "temp"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

VOICE_MAP = {"hi-IN": "anushka", "en-IN": "anushka", "te-IN": "anushka", "ta-IN": "anushka"}
DEFAULT_VOICE = "anushka"

HOSPITAL_NAME = os.getenv("HOSPITAL_NAME", "Sunrise Hospital")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "NyraAI")

try:
    sarvam_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize AI clients: {e}")

logging.basicConfig(level=logging.INFO)

# -----------------------
# Dummy Booking Data
# -----------------------
DUMMY_AVAILABLE_SLOTS = {
    "Dr. Sharma": {
        "department": "Cardiology",
        "available": [
            {"date": "2025-11-06", "time": "10:00 AM"},
            {"date": "2025-11-06", "time": "11:00 AM"},
            {"date": "2025-11-07", "time": "03:00 PM"},
        ]
    },
    "Dr. Gupta": {
        "department": "General Medicine",
        "available": [
            {"date": "2025-11-06", "time": "09:30 AM"},
            {"date": "2025-11-06", "time": "12:00 PM"},
            {"date": "2025-11-08", "time": "02:00 PM"},
        ]
    }
}


# -----------------------
# Helpers (No changes to extract_json_from_text or remove_json_and_metadata_markers)
# -----------------------
def extract_json_from_text(text: str):
    """Return first balanced JSON object parsed from text, or None."""
    if not text:
        return None
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in starts:
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    cand = text[s : i + 1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        break
    return None


def remove_json_and_metadata_markers(text: str):
    """
    Remove <METADATA>...</METADATA> and first balanced JSON block.
    Return (cleaned_text, removed_parts_list).
    """
    removed = []
    if not text:
        return text, removed

    # remove <METADATA> ... </METADATA>
    meta_pattern = re.compile(r"(?is)<metadata>(.*?)</metadata>")
    def _meta_repl(m):
        removed.append(m.group(0))
        return " "
    text = meta_pattern.sub(_meta_repl, text)

    # remove first balanced JSON block (search from end)
    starts = [m.start() for m in re.finditer(r"\{", text)]
    if starts:
        for s in reversed(starts):
            depth = 0
            for i in range(s, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        cand = text[s:i+1]
                        try:
                            json.loads(cand)
                            removed.append(cand)
                            text = text[:s] + " " + text[i+1:]
                            raise StopIteration
                        except StopIteration:
                            break
                        except Exception:
                            break
            else:
                continue
            break

    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned, removed


async def llm_for_response(user_transcript: str, language_code: str, request_id: str, history: dict, available_slots: dict):
    """
    Call LLM to produce only the natural-language reply (no metadata).
    This LLM is now stateful and aware of booking capabilities.
    """
    if not user_transcript:
        return None
        
    history_json = json.dumps(history, ensure_ascii=False, indent=2)
    slots_json = json.dumps(available_slots, ensure_ascii=False, indent=2)
    
    system_prompt = f"""
You are {ASSISTANT_NAME}, the friendly, empathetic, and highly competent virtual assistant for {HOSPITAL_NAME}.
You MUST reply primarily in the language identified by the code '{language_code}'.

Initial greeting (say once at conversation start):
- Say a single short localized greeting that mentions the agent and hospital, using the user's language ({language_code}).
  Example for Telugu: "Naira AI agent {HOSPITAL_NAME} తరపున మీతో మాట్లాడుతోంది."
  Do NOT repeat this greeting again later in the same conversation.

Tone & scope:
- Sound human, warm, natural, and empathetic (e.g., "I'm sorry to hear you're not feeling well — let's get this sorted." translated into {language_code}).
- Keep replies short and conversational.
- Focus strictly on hospital tasks: booking, rescheduling, cancelling appointments, doctor availability, general hospital info, emergency triage, language switch, or transfer to a human agent.
- Do not answer unrelated questions except a brief clarifying question when needed.

Language mixing rule:
- Reply primarily in {language_code}.
- Short, natural insertions of English words are allowed when they make the reply more natural (examples: "available ga unnaru", "slot confirm cheyali", "mobile number"); however, prefer phrases and sentence structure in {language_code}.
- Avoid full English sentences when the conversation is in {language_code}.

Doctor list & specialty rule:
- When listing doctors, present each on a separate plain line with the specialty shown in English only.
  Example:
  Dr. Gupta — General Medicine
  Dr. Sharma — Cardiology
- If the user asks what a specialty means, explain the specialty briefly in the user's language ({language_code}), one short sentence.

Conversation rules and slot handling (strict):
1. Read the "User's Latest Transcript" and respond naturally in {language_code}.
2. Be empathetic. If the user is unwell, acknowledge kindly.
3. ALWAYS STATE FULL SLOT: When offering a slot, always state the full date and full time (e.g., "Dr. Gupta has a slot on November 6, 2025 at 9:30 AM"), but translate the surrounding sentence into {language_code} (you may keep the date/time formatted in English numerals).
4. ASK FOR MISSING DETAILS: Mandatory.
   - After the user agrees to a slot, ask for any missing confirmation details needed to finalize the booking (mobile number, name, reason for visit, preferred location/branch).
   - Example (in {language_code}): "Great — Dr. Gupta on November 6, 2025 at 9:30 AM available gaa undi. Lock cheyadaniki mee mobile number and reason for the visit kavali."
5. Always ask for `reason` (if unknown) and `location` (if not provided).
6. NEVER append JSON or metadata in replies — only natural conversational text.
7. Keep replies short and focused on the next actionable step.

Error / repetition rule:
- Do not repeat the initial greeting more than once.
- Avoid excessive punctuation, asterisks, brackets, or formatting — keep output plain and user-friendly.

Available Slots and Conversation History will be injected as:
Conversation history (so far):
{history_json}

Available Slots (the only slots you can book):
{slots_json}
"""



    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User's Latest Transcript: {user_transcript}"},
            ],
            stream=False,
        )
        # defensive extraction
        try:
            choice0 = resp.choices[0]
            if hasattr(choice0, "message") and getattr(choice0.message, "content", None):
                return choice0.message.content.strip()
            elif isinstance(choice0, dict):
                m = choice0.get("message", {})
                return (m.get("content") or choice0.get("text") or "").strip()
            else:
                return (getattr(choice0, "text", "") or "").strip()
        except Exception:
            return str(resp).strip()
    except Exception as e:
        logging.exception(f"[{request_id}] llm_for_response failed: {e}")
        return None


async def llm_for_json(user_transcript: str, llm_text: str, request_id: str, history: dict):
    """
    Call LLM to return STRICT JSON representing the *new updated state*.
    The model must return only JSON. Returns dict or None.
    """
    
    # We only need the state, not the full history object (which includes conversations)
    previous_state = {
        "user_details": history.get("user_details", {}),
        "bookings": history.get("bookings", [])
    }
    previous_state_json = json.dumps(previous_state, ensure_ascii=False, indent=2)

    system_prompt = f"""
You are a JSON-only state updater. Given the previous conversation state (`previous_state`), the user's latest transcript (`user_transcript`), and the assistant's reply (`llm_reply`), your job is to return the **ENTIRELY NEW** state object, structured according to the schema.

**Do not just return the *changes***. Return the **full, updated** `user_details` and `bookings` objects as they should now exist.

**Previous State:**
{previous_state_json}

**Current Interaction:**
User: "{user_transcript}"
Assistant: "{llm_text}"

**Instructions:**
1.  Analyze the "Current Interaction".
2.  **Update `user_details`**: Take the `user_details` from the "Previous State" and merge any *new* information from the "Current Interaction" (e.g., name, phone, location, language).
3.  **Update `bookings`**:
    * Take the `bookings` array from the "Previous State".
    * If the "Current Interaction" *confirms* a **new** booking, add a new booking object to the array. When adding a new booking, generate a **new UUID** for the `appointment_id` and set `status` to "confirmed".
    * If the "Current Interaction" *cancels* or *reschedules* an **existing** booking, update its `status` or details in the array.
4.  Return **ONLY** the JSON for the *updated state* in this exact format:
{{
  "user_details": {{
    "name": "",
    "age": "",
    "phone": "",
    "language": "",
    "location": ""
  }},
  "bookings": [
    {{
      "doctor_name": "",
      "department": "",
      "date": "",
      "time": "",
      "reason": "",
      "appointment_id": "<existing_or_new_uuid>",
      "status": "<confirmed|cancelled|rescheduled>"
    }}
  ]
}}

Only output the JSON object and nothing else. Use null or empty strings for unknown fields.
Generate a new `appointment_id` using `str(uuid.uuid4())` ONLY for newly confirmed bookings.
"""
    user_prompt = f"Previous State:\n{previous_state_json}\n\nUser transcript:\n{user_transcript}\n\nAssistant reply:\n{llm_text}\n\nProduce the updated state JSON now."

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
        # get raw text
        raw = ""
        try:
            ch0 = resp.choices[0]
            if hasattr(ch0, "message") and getattr(ch0.message, "content", None):
                raw = ch0.message.content
            elif isinstance(ch0, dict):
                raw = ch0.get("message", {}).get("content") or ch0.get("text") or ""
            else:
                raw = getattr(ch0, "text", "") or ""
        except Exception:
            raw = str(resp)

        raw = (raw or "").strip()
        # parse
        try:
            parsed = json.loads(raw)
            return parsed
        except Exception:
            parsed = extract_json_from_text(raw)
            return parsed
    except Exception as e:
        logging.exception(f"[{request_id}] llm_for_json failed: {e}")
        return None


async def synthesize_and_send_tts(websocket: WebSocket, text: str, language_code: str, request_id: str):
    """
    Convert text to audio using Sarvam streaming TTS (short-lived session),
    collect audio bytes, and send to websocket in one binary blob (audio_start -> binary -> audio_end).
    """
    if not text or not text.strip():
        return False

    audio_bytes_buffer = bytearray()
    try:
        async with sarvam_client.text_to_speech_streaming.connect(model="bulbul:v2", send_completion_event=True) as tts_ws:
            await tts_ws.configure(target_language_code=language_code, speaker=VOICE_MAP.get(language_code, DEFAULT_VOICE))
            try:
                await tts_ws.convert(text)
                await tts_ws.flush()
            except Exception as e:
                logging.exception(f"[{request_id}] TTS convert/flush error: {e}")

            try:
                async for msg in tts_ws:
                    if isinstance(msg, AudioOutput):
                        try:
                            chunk = base64.b64decode(msg.data.audio)
                            audio_bytes_buffer.extend(chunk)
                        except Exception:
                            logging.debug(f"[{request_id}] failed decode audio chunk")
                        continue
                    if isinstance(msg, EventResponse):
                        ev = getattr(msg.data, "event_type", None)
                        logging.debug(f"[{request_id}] TTS event: {ev}")
                        if ev == "final":
                            break
                    # defensive: check msg.data.audio
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
                logging.info(f"[{request_id}] tts collection ended: {e}")
    except Exception as e:
        logging.exception(f"[{request_id}] failed to open tts ws: {e}")

    # send to client
    if audio_bytes_buffer:
        try:
            await websocket.send_json({"type": "audio_start"})
        except Exception:
            logging.debug(f"[{request_id}] couldn't send audio_start")
        try:
            await websocket.send_bytes(bytes(audio_bytes_buffer))
            logging.info(f"[{request_id}] Sent audio blob to client ({len(audio_bytes_buffer)} bytes)")
        except Exception:
            logging.exception(f"[{request_id}] Failed to send binary blob")
        try:
            await websocket.send_json({"type": "audio_end", "bytes": len(audio_bytes_buffer)})
        except Exception:
            logging.debug(f"[{request_id}] couldn't send audio_end")
        # debug write
        try:
            debug_path = os.path.join(os.getcwd(), "debug_last.mp3")
            with open(debug_path, "wb") as df:
                df.write(bytes(audio_bytes_buffer))
        except Exception:
            pass
        return True
    else:
        logging.warning(f"[{request_id}] No audio bytes collected from TTS")
        return False


# -----------------------
# Main pipeline
# -----------------------
async def run_batch_stt_pipeline(audio_file_path: str, request_id: str, websocket: WebSocket):
    """
    1) Load History
    2) Batch STT
    3) llm_for_response (stateful) -> speak (TTS)
    4) llm_for_json (stateful) -> get updated state
    5) Append turn to `conversations`
    6) Save new state to data/<request_id>.json
    """
    try:
        # 1) Load History or create new structure
        history_path = os.path.join(DATA_DIR, f"{request_id}.json")
        history = {
            "requestId": request_id,
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "conversations": [],
            "user_details": {"name": "", "age": "", "phone": "", "language": "", "location": ""},
            "bookings": []
        }
        try:
            if os.path.exists(history_path):
                with open(history_path, "r", encoding="utf-8") as fh:
                    history = json.load(fh)
                    # Ensure all keys are present
                    if "conversations" not in history: history["conversations"] = []
                    if "user_details" not in history: history["user_details"] = {"name": "", "age": "", "phone": "", "language": "", "location": ""}
                    if "bookings" not in history: history["bookings"] = []
        except Exception as e:
            logging.debug(f"[{request_id}] could not read existing history; creating new: {e}")


        # 2) Batch STT
        logging.info(f"[{request_id}] Starting BATCH STT with file: {audio_file_path}...")
        user_transcript = ""
        language_code = "te-IN"

        try:
            with open(audio_file_path, "rb") as f:
                stt_resp = await sarvam_client.speech_to_text.transcribe(file=f, model="saarika:v2.5")
            logging.info(f"[{request_id}] STT raw: {stt_resp}")
            if getattr(stt_resp, "transcript", None):
                user_transcript = stt_resp.transcript
                language_code = getattr(stt_resp, "language_code", language_code) or language_code
            else:
                await websocket.send_json({"type": "error", "message": "Could not transcribe."})
                return
        except Exception as e:
            logging.exception(f" STT error: {e}")
            await websocket.send_json({"type": "error", "message": "STT failed."})
            return

        # send transcript to client
        try:
            await websocket.send_json({"type": "final_transcript", "text": user_transcript})
        except Exception:
            pass

        logging.info(f"[{request_id}] User said: {user_transcript} (lang={language_code})")
        
        # Update language in state
        if "user_details" in history and history["user_details"] is not None:
            history["user_details"]["language"] = language_code
        else:
             history["user_details"] = {"language": language_code}


        # 3) LLM for response (stateful, with history and slots)
        llm_response_text = await llm_for_response(user_transcript, language_code, request_id, history, DUMMY_AVAILABLE_SLOTS)
        if not llm_response_text:
            llm_response_text = "సారీ, నేను మీకో ఒక సమాధానం ఇవ్వలేకపోతున్నాను."  # fallback friendly msg in telugu
        
        # send llm text to client for display
        try:
            await websocket.send_json({"type": "llm_response", "text": llm_response_text})
        except Exception:
            pass

        # Synthesize & send TTS
        await synthesize_and_send_tts(websocket, llm_response_text, language_code, request_id)

        # 4) LLM for strict JSON state update
        updated_state = await llm_for_json(user_transcript, llm_response_text, request_id, history)

        # 5) Append turn to `conversations`
        turn = {
            "user": user_transcript,
            "llm": llm_response_text,
            "createdAt": datetime.utcnow().isoformat() + "Z",
        }
        history["conversations"].append(turn)

        # 6) Update state from llm_for_json
        if updated_state:
            if "user_details" in updated_state and updated_state["user_details"] is not None:
                history["user_details"] = updated_state["user_details"]
            if "bookings" in updated_state and updated_state["bookings"] is not None:
                history["bookings"] = updated_state["bookings"]
        else:
            logging.warning(f"[{request_id}] llm_for_json returned None; state not updated for this turn.")
            
        history["lastUpdatedAt"] = datetime.utcnow().isoformat() + "Z"

        # 7) send updated state to client
        try:
            await websocket.send_json({"type": "session_update", "state": history})
        except Exception:
            pass

        # 8) Save structured history
        try:
            with open(history_path, "w", encoding="utf-8") as fh:
                json.dump(history, fh, ensure_ascii=False, indent=2)
            logging.info(f"[{request_id}] Saved updated history to {history_path}")
        except Exception:
            logging.exception(f"[{request_id}] failed to write history")

    except Exception as e:
        logging.exception(f"[{request_id}] pipeline error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": "Pipeline error."})
        except Exception:
            pass


# -----------------------
# WebSocket endpoint
# -----------------------
# -----------------------
# WebSocket endpoint
# -----------------------
@router.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket):
    await websocket.accept()
    request_id = str(uuid.uuid4())
    logging.info(f"[{request_id}] WS connected")

    audio_buffer = []
    temp_files = []
    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                logging.info(f"[{request_id}] client disconnected (inside loop)")
                break

            if "bytes" in message:
                audio_buffer.append(message["bytes"])
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                except Exception:
                    logging.warning(f"[{request_id}] Received non-json text")
                    continue

                if data.get("type") == "stop_speaking":
                    if not audio_buffer:
                        logging.warning(f"[{request_id}] stop_speaking with empty buffer")
                        continue
                    
                    # --- FIX: Generate a unique ID for *this turn's* audio file ---
                    turn_id = str(uuid.uuid4())
                    input_audio_path = os.path.join(TEMP_DIR, f"{request_id}_{turn_id}_input.webm")
                    # --- END FIX ---
                    
                    temp_files.append(input_audio_path)
                    with open(input_audio_path, "wb") as f:
                        for chunk in audio_buffer:
                            f.write(chunk)
                    audio_buffer = []
                    # run pipeline synchronously
                    await run_batch_stt_pipeline(input_audio_path, request_id, websocket)
    except WebSocketDisconnect:
        logging.info(f"[{request_id}] client disconnected (outside loop)")
    except Exception as e:
        logging.exception(f"[{request_id}] WS main error: {e}")
    finally:
        logging.info(f"[{request_id}] cleaning up {len(temp_files)} files")
        for p in temp_files:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    await websocket.accept()
    request_id = str(uuid.uuid4())
    logging.info(f"[{request_id}] WS connected")

    audio_buffer = []
    temp_files = []
    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                logging.info(f"[{request_id}] client disconnected (inside loop)")
                break

            if "bytes" in message:
                audio_buffer.append(message["bytes"])
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                except Exception:
                    logging.warning(f"[{request_id}] Received non-json text")
                    continue

                if data.get("type") == "stop_speaking":
                    if not audio_buffer:
                        logging.warning(f"[{request_id}] stop_speaking with empty buffer")
                        continue
                    # save audio to file
                    input_audio_path = os.path.join(TEMP_DIR, f"{request_id}_input.webm")
                    temp_files.append(input_audio_path)
                    with open(input_audio_path, "wb") as f:
                        for chunk in audio_buffer:
                            f.write(chunk)
                    audio_buffer = []
                    # run pipeline synchronously
                    await run_batch_stt_pipeline(input_audio_path, request_id, websocket)
    except WebSocketDisconnect:
        logging.info(f"[{request_id}] client disconnected (outside loop)")
    except Exception as e:
        logging.exception(f"[{request_id}] WS main error: {e}")
    finally:
        logging.info(f"[{request_id}] cleaning up {len(temp_files)} files")
        for p in temp_files:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass