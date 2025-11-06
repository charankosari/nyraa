# agent.py
import os
import re
import json
import uuid
import logging
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
HOSPITAL_NAME_TELUGU = os.getenv("HOSPITAL_NAME_TELUGU","సన్రైస్ హాస్పిటల్")  # e.g. "సన్రైస్ హాస్పిటల్"
ASSISTANT_NAME_TELUGU = os.getenv("ASSISTANT_NAME_TELUGU","నైరా ఏ ఐ ఏజెంట్")  # e.g. "నైరా ఏఐ ఏజెంట్"
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
# Booking helper stubs
# -----------------------
def book_appointment(data: dict):
    """
    Stub for creating a booking in your real system.
    For now: just print the booking data. Replace with DB/API call later.
    """
    try:
        print("[BOOK_APPOINTMENT] Booking data:", json.dumps(data, ensure_ascii=False))
    except Exception:
        print("[BOOK_APPOINTMENT] Booking data (could not JSON dump):", data)


def cancel_booking(data: dict):
    """
    Stub for cancelling a booking in your real system.
    Expected data contains appointment_id and optionally reason.
    """
    try:
        print("[CANCEL_BOOKING] Cancel request:", json.dumps(data, ensure_ascii=False))
        # simulate cancel result
        print(f"[CANCEL_BOOKING] appointment {data.get('appointment_id')} cancelled.")
    except Exception:
        print("[CANCEL_BOOKING] Cancel data:", data)


def reschedule_booking(data: dict):
    """
    Stub for rescheduling a booking.
    data should contain appointment_id and new date/time.
    """
    try:
        print("[RESCHEDULE_BOOKING] Reschedule request:", json.dumps(data, ensure_ascii=False))
        print(f"[RESCHEDULE_BOOKING] appointment {data.get('appointment_id')} rescheduled.")
    except Exception:
        print("[RESCHEDULE_BOOKING] Reschedule data:", data)

# -----------------------
# Intent / prompt helpers (AI-driven)
# -----------------------
# -----------------------
# Intent / prompt helpers (overwrite mode)
# -----------------------
def set_additional_prompt(history: dict, key: str, value):
    """
    Store one small, structured metadata item in history['additional_prompts_map'].
    - This overwrites previous values for the same key (no repeated list entries).
    - We keep a map for quick lookup, but also preserve a concise list for ordering.
    """
    if not history:
        return
    # maintain both a map (for overwrite) and a short list (for chronological audit if needed)
    if "additional_prompts_map" not in history or history["additional_prompts_map"] is None:
        history["additional_prompts_map"] = {}
    if "additional_prompts" not in history or history["additional_prompts"] is None:
        history["additional_prompts"] = []

    ts = datetime.utcnow().isoformat() + "Z"
    # overwrite map entry
    history["additional_prompts_map"][key] = {"at": ts, "v": value}

    # also update the short list: remove any existing entry with same key, then append
    # list entries keep small objects {k, at, v} for UI/audit but are unique per key
    new_entry = {"k": key, "at": ts, "v": value}
    # remove previous same-key entry if present
    history["additional_prompts"] = [e for e in history["additional_prompts"] if e.get("k") != key]
    history["additional_prompts"].append(new_entry)


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
    greeting_sent = False
    try:
        # prefer explicit flag if present
        if history.get("_greeting_sent"):
            greeting_sent = True
        else:
            for c in history.get("conversations", []):
                llm_msg = (c.get("llm") or "")
                if "నైరా" in llm_msg or (ASSISTANT_NAME_TELUGU and ASSISTANT_NAME_TELUGU in llm_msg):
                    greeting_sent = True
                    break
    except Exception:
        greeting_sent = False

    # ----- new: prepare slots_text that the LLM can read (or a "no slots" message) -----
    # ----- formatting helpers for TTS-friendly slots (date -> "Month D, YYYY", time -> "9 AM" or "9:30 AM") -----
    def _format_time_for_tts(t: str):
        # t examples: "09:30 AM", "9:30 AM", "12:00 PM", "03:00 PM"
        if not t:
            return ""
        t = t.strip()
        # remove leading zero from hour
        t = re.sub(r"^0", "", t)
        # normalize AM/PM spacing and uppercase
        t = re.sub(r"\s+", " ", t).upper()
        # if hour:00, convert to "9 AM" style
        m = re.match(r"^(\d{1,2}):00\s*(AM|PM)$", t, flags=re.I)
        if m:
            return f"{int(m.group(1))} {m.group(2).upper()}"
        return t

    def _format_date_for_tts(date_iso: str):
        # Convert YYYY-MM-DD to "Month D, YYYY" (e.g., 2025-11-06 -> November 6, 2025)
        try:
            dt = datetime.strptime(date_iso, "%Y-%m-%d")
            return dt.strftime("%B %-d, %Y")  # on Linux/Unix the %-d removes leading zero
        except Exception:
            # fallback: try without %-d for Windows compatibility
            try:
                dt = datetime.strptime(date_iso, "%Y-%m-%d")
                return dt.strftime("%B %d, %Y")
            except Exception:
                return date_iso

    slots_text_lines = []
    if not available_slots:
        slots_text = "No available slots."
    else:
        for doc, meta in available_slots.items():
            dept = meta.get("department", "")
            av = meta.get("available", []) or []
            if not av:
                slots_text_lines.append(f"{doc} — {dept}: NO_SLOTS")
                continue
            slot_parts = []
            for s in av:
                d_iso = s.get("date", "")
                d = _format_date_for_tts(d_iso) if d_iso else ""
                tm = _format_time_for_tts(s.get("time", ""))
                if d and tm:
                    slot_parts.append(f"{d} at {tm}")
                elif d:
                    slot_parts.append(d)
                elif tm:
                    slot_parts.append(tm)
            slots_text_lines.append(f"{doc} — {dept}: " + ", ".join(slot_parts))
        slots_text = "\n".join(slots_text_lines) if slots_text_lines else "No available slots."


    # small greeting clause to inject into the prompt (model should only speak greeting if greeting_sent is False)
    if greeting_sent:
        greeting_clause = "History indicates the initial greeting has already been sent; DO NOT repeat the initial greeting."
    else:
        # prefer Telugu names if provided
        agent_name_telugu = ASSISTANT_NAME_TELUGU or ASSISTANT_NAME
        hosp_name_telugu = HOSPITAL_NAME_TELUGU or HOSPITAL_NAME
        greeting_clause = (
            f'At the start of the conversation say a single short localized greeting that mentions the agent and hospital '
            f'in Telugu script (example): \"{agent_name_telugu} — {hosp_name_telugu} తరపున మీతో మాట్లాడుతున్నది.\" '
            "ONLY say this greeting if it has not already appeared in history."
        )

    # recompute history_json for prompt (safe string)
    history_json = json.dumps(history, ensure_ascii=False, indent=2)
    slots_json = json.dumps(available_slots, ensure_ascii=False, indent=2)
    
    system_prompt = f"""
You are {ASSISTANT_NAME}, an empathetic, professional hospital assistant for {HOSPITAL_NAME}.
You MUST reply primarily in the language identified by the code '{language_code}' and you MUST keep replies short and action-oriented.
# ACTIVE-INTENT RULES (CRITICAL)
- This conversation may have an existing active intent in the session state. The assistant must **stay within the active intent** until it is explicitly completed or cancelled by the user.
- If history.active_intent is non-empty, DO NOT change the intent to another action. Continue the active flow (ask exactly one next question if missing data, confirm proposed changes, or apply the requested action when the user confirms).
- Only clear the active intent when:
  * The booking status becomes a final state: "confirmed", "rescheduled", or "cancelled", OR
  * The user explicitly asks to stop/cancel the flow or switch intent (explicit phrases like "stop", "cancel", "I want to do something else", "book new appointment").
- When an intent is started for the first time (no active_intent), you may set active_intent to one of: intent_book_appointment, intent_reschedule, intent_cancel — but once set, keep it until finished.
- Ask **one** question at a time (one missing field per reply). Wait for user's answer before asking another. If all required fields are present, perform the action (confirm the booking/reschedule/cancel) and produce a short confirmation sentence.
# MANDATORY FIELDS / PROGRESS
- Required booking fields: name, phone, reason, location, language, emergency_status.
- Use the TTS-friendly time/date formatting rules (as before).
- When suggesting or confirming a slot, say exactly the final TTS-friendly slot string (e.g. "November 6, 2025 at 9 AM") so the user can confirm.

GREETING:
- {greeting_clause}
- If the user's language code is 'te-IN', prefer the Telugu-script names: "{ASSISTANT_NAME_TELUGU}" and "{HOSPITAL_NAME_TELUGU}". Otherwise use the ASCII names "{ASSISTANT_NAME}" and "{HOSPITAL_NAME}".

LANGUAGE RULES:
- Detect and reply in the user's language (use the provided language code). Use polite, human phrasing.
- Very short English insertions are allowed, but prefer native-language phrasing and sentence structure.
- When switching languages (user asked to change), confirm the language change and then reply in the new language from that turn onward.
- Dont make any mistakes like మిమ్మల్ని ఎలా సహాయపడగలను instead మికు ఎలా సహాయపడగలను 
SCOPE & TONE:
- Stay strictly within hospital-related tasks: booking, rescheduling, cancelling appointments, checking doctor availability, triage/emergency status, language switch, or transferring to human.
- Be empathetic and professional. Keep output short (1–3 short sentences or 2 short bullets).
- Avoid extra punctuation, emojis, markup, or code. Output plain text only.

DOCTOR LISTING FORMAT:
- When listing doctors present each on a separate plain line and list specialty in English only:
  Dr. Gupta — General Medicine
  Dr. Sharma — Cardiology

DOCTORS AVAILABLE SLOTS (TTS-friendly):
{slots_text}


TIME / TTS-FRIENDLY FORMATTING (IMPORTANT):
- When giving dates/times, format them exactly like:
  "November 6, 2025 at 9 AM"
  - Do NOT use colons or special punctuation in times (avoid "9:00 AM" — use "9 AM").
  - Use English numerals for clarity.
  - Keep the surrounding sentence in the user's language.
- This prevents the TTS engine from producing unnatural speech.

MANDATORY DATA FIELDS & ASKING FOR MISSING DATA:
- Always ask for missing mandatory booking info before confirming:
  * name, phone number, reason for visit, preferred location/branch, language (if unknown), and emergency_status (choose one: low, medium, high).
- Ask one question at a time. Example:
  "Great — I can book Dr. Gupta on November 6, 2025 at 9 AM. To confirm, please give your full name."
- If user gives partial info, acknowledge and ask the remaining required fields.

BOOKING / RESCHEDULE / CANCEL WORKFLOW (strict):
- Booking:
  * Offer only available slots from the provided Available Slots.
  * When user confirms a slot, ask for any missing mandatory fields (name, phone, reason, location, emergency_status).
  * After collecting required fields, respond with a brief confirmation sentence (in user's language) and the exact full slot text (date and time formatted per TTS rule).
  * Do NOT output or append any JSON or metadata to the user's reply (plain text only).
- Rescheduling:
  * Require the existing `appointment_id` to start reschedule flow.
  * Ask the user for their preferred new time and propose available alternatives if the requested time is unavailable.
  * Ask confirmation: "Is this new time OK?" If user says yes, update the booking and indicate the booking was rescheduled.
  * Save `emergency_status` if provided or ask for it.
- Cancelling:
  * Require the existing `appointment_id`.
  * Ask for explicit confirmation to cancel (e.g., "Please confirm you want to cancel appointment <ID> — say 'yes' to cancel').
  * After explicit confirmation, mark booking cancelled and respond with a brief cancellation confirmation.

EMERGENCY / TRIAGE:
- If user describes urgent symptoms, ask quick triage questions and set `emergency_status` to "high" if life-threatening signs present.
- Always advise to call emergency services immediately in life-threatening situations (give a short phrase in user's language).

ENDING THE FLOW:
- After completing a booking, reschedule, or cancellation, politely ask if the user needs further assistance.
- If user says "no" or similar, end with a polite closing sentence.
- THEN SAY GOODBYE and have a nice day in the user's language.
STATE & JSON RULES (for internal state updater model):
- llm_for_response must return **only** natural language reply (no JSON, no metadata).
- llm_for_json (the state-updater model) must return only **JSON** (the entire updated `user_details` and `bookings` object) per the schema provided to the system.

ERRORS & CLARIFICATIONS:
- If user input is unclear, ask a single short clarifying question.
- If a user provides a booking id that doesn't exist, say a short message asking them to re-check the ID.

VOICE / NAME SPEECH:
- The agent & hospital names must appear in the initial greeting in Telugu script.
- When repeating the names later, you may use the user's language but prefer using the Telugu script at least on first mention.

EXAMPLES (formatting guidance):
- Confirmed booking reply example (Telugu):
  "బుక్ అయింది: Dr. Gupta — General Medicine, November 6, 2025 at 9 AM. నేను మీకు ఒక కన్ఫర్మేషన్ SMS పంపిస్తాను. నాకు మీ ఫోన్ నంబర్ చెప్పగలరా?"
- Cancel confirmation request example:
  "మీరు appointment-id 12345 ను రద్దు చేయాలనుకుంటున్నారా? ధృవీకరించడానికి 'yes' లేదా 'no' చెప్పండి."

STRICT: No JSON or metadata in llm_for_response outputs. Keep replies short and actionable.
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


async def llm_for_json(user_transcript: str, llm_text: str, request_id: str, history: dict,language_code: str):
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
You are a JSON-only state updater. Given the previous conversation state (`previous_state`) {previous_state_json}, the user's latest transcript (`user_transcript`) {user_transcript}, and the assistant's reply (`llm_reply`) {llm_text}, your job is to return the **ENTIRELY NEW** state object, structured according to the schema.

***ACTIVE-INTENT / PROGRESS RULES (MUST FOLLOW)***
- If `previous_state.active_intent` is non-empty, you MUST keep `intent` equal to that value **unless** the user explicitly asked to switch or cancel the flow in their transcript.
- Use booking `status` values to represent progress:
  * For new booking flows not yet confirmed, use `"pending"`.
  * For reschedule flows in progress, use `"reschedule_in_progress"`.
  * When the user confirms the reschedule, set status to `"rescheduled"`.
  * When the user confirms a new booking, set status to `"confirmed"` and include `appointment_id` (generate with `str(uuid.uuid4())`).
  * For cancellations: when user confirms cancellation, set status to `"cancelled"`.
- **Only** when status is `"confirmed"`, `"rescheduled"`, or `"cancelled"` should downstream code call booking/cancel/reschedule stubs. In-progress statuses indicate the flow is still gathering info.
- If fields are missing, include them as empty strings in the bookings object and ensure `user_details` contains partial data if provided.

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
  "intent": "<intent_name>",  # one of: intent_book_appointment, intent_reschedule, intent_cancel, intent_doctor_availability, intent_general_info, intent_emergency, intent_language_switch, intent_transfer_to_human,
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
    active_intent = history.get("active_intent", "")
    pending_id = history.get("pending_appointment_id", "")
    pending_fields = history.get("pending_required_fields", {})
    state_snippet = json.dumps({
        "active_intent": active_intent,
        "pending_appointment_id": pending_id,
        "pending_required_fields": pending_fields,
    }, ensure_ascii=False)
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"State: {state_snippet}\nUser's Latest Transcript: {user_transcript}"},
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
        "bookings": [],
        "reschedule": [],   # new: store reschedule events (small structs)
        "cancel": [],       # new: store cancel events (small structs)
        "additional_prompts": [],      # short unique list
        "additional_prompts_map": {},  # map for overwrite semantics
         "active_intent": "",               # one of: intent_book_appointment, intent_reschedule, intent_cancel, "" (empty when none)
    "pending_appointment_id": "",      # appointment_id we're currently acting on (if any)
    "pending_required_fields": {},     # map of which fields still required (e.g., {"name": True, "phone": True})
        }

        try:
            if os.path.exists(history_path):
                with open(history_path, "r", encoding="utf-8") as fh:
                    history = json.load(fh)
                    if "_greeting_sent" not in history:
                        history["_greeting_sent"] = False
                    # Ensure all keys are present
                    if "conversations" not in history: history["conversations"] = []
                    if "user_details" not in history: history["user_details"] = {"name": "", "age": "", "phone": "", "language": "", "location": ""}
                    if "bookings" not in history: history["bookings"] = []
                    if "active_intent" not in history: history["active_intent"] = ""
                    if "pending_appointment_id" not in history: history["pending_appointment_id"] = ""
                    if "pending_required_fields" not in history: history["pending_required_fields"] = {}

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
        try:
            if ("నైరా" in llm_response_text) or (ASSISTANT_NAME_TELUGU and ASSISTANT_NAME_TELUGU in llm_response_text):
                history["_greeting_sent"] = True
        except Exception:
            pass
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
        updated_state = await llm_for_json(user_transcript, llm_response_text, request_id, history,language_code)
        # -----------------------
        # Capture model-decided intent or action
        # -----------------------
       # -----------------------
        # Capture model-decided intent or action (overwrite existing)
        # -----------------------
        try:
            intent_field = None
            if isinstance(updated_state, dict):
                intent_field = (
                    updated_state.get("intent")
                    or updated_state.get("user_intent")
                    or updated_state.get("intent_type")
                    or updated_state.get("action")
                )

            if intent_field:
                set_additional_prompt(history, "intent", intent_field)
                appt_id = None
                bks = updated_state.get("bookings", []) if isinstance(updated_state.get("bookings", []), list) else []
                if bks:
                    first = bks[0]
                    appt_id = first.get("appointment_id") or first.get("appointmentId") or first.get("id")
                if not appt_id:
                    appt_id = history.get("pending_appointment_id") or ""
                event = {"at": datetime.utcnow().isoformat() + "Z", "intent": intent_field, "appointment_id": appt_id}
                if "resched" in (intent_field or "").lower() or "reschedule" in (intent_field or "").lower() or "intent_reschedule" == intent_field:
                    history.setdefault("reschedule", []).append(event)
                elif "cancel" in (intent_field or "").lower() or "intent_cancel" == intent_field:
                    history.setdefault("cancel", []).append(event)
            else:
                set_additional_prompt(history, "intent", "unspecified")
        except Exception as e:
            logging.warning(f"[{request_id}] Failed to log model intent (overwrite): {e}")


        # 5) Append turn to `conversations`
        turn = {
            "user": user_transcript,
            "llm": llm_response_text,
            "createdAt": datetime.utcnow().isoformat() + "Z",
        }
        history["conversations"].append(turn)

        # 6) Update state from llm_for_json
        if updated_state:
            prev_bookings = history.get("bookings", [])
            prev_map = {b.get("appointment_id"): b for b in prev_bookings if b.get("appointment_id")}
            new_bookings = updated_state.get("bookings", []) or []

            # Detect newly confirmed bookings (appointment_id not present before)
            for b in new_bookings:
                aid = b.get("appointment_id")
                status = b.get("status", "")
                if aid and aid not in prev_map and status == "confirmed":
                    # new booking - call stub
                    book_appointment(b)

            # Detect cancellations / status changed
            for b in new_bookings:
                aid = b.get("appointment_id")
                status = b.get("status", "")
                if aid and aid in prev_map:
                    prev_status = prev_map[aid].get("status", "")
                    if prev_status != status:
                        if status == "cancelled":
                            cancel_booking(b)
                        elif status == "rescheduled":
                            reschedule_booking(b)

            # Overwrite history with updated_state values
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
            except (WebSocketDisconnect, RuntimeError) as e:
                logging.info(f"[{request_id}] client disconnected (inside loop) or receive failed: {e}")
                break


            # collect binary audio chunks
            if "bytes" in message:
                audio_buffer.append(message["bytes"])
                continue

            # handle control/text frames
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except Exception:
                    logging.warning(f"[{request_id}] Received non-json text")
                    continue

                if data.get("type") == "stop_speaking":
                    if not audio_buffer:
                        logging.warning(f"[{request_id}] stop_speaking with empty buffer")
                        continue

                    # --- create a unique temp file for this turn ---
                    turn_id = str(uuid.uuid4())
                    input_audio_path = os.path.join(TEMP_DIR, f"{request_id}_{turn_id}_input.webm")

                    # write the in-memory chunks to disk
                    try:
                        temp_files.append(input_audio_path)
                        with open(input_audio_path, "wb") as f:
                            for chunk in audio_buffer:
                                f.write(chunk)
                    except Exception as e:
                        logging.exception(f"[{request_id}] failed to write temp audio file: {e}")
                        audio_buffer = []
                        # remove from temp_files if append succeeded but write failed
                        try:
                            temp_files.remove(input_audio_path)
                        except Exception:
                            pass
                        continue

                    # clear the in-memory buffer (we now have the file)
                    audio_buffer = []

                    # run the pipeline (STT -> LLM -> TTS -> state update)
                    try:
                        await run_batch_stt_pipeline(input_audio_path, request_id, websocket)
                    except Exception as e:
                        logging.exception(f"[{request_id}] run_batch_stt_pipeline failed: {e}")

                    # remove the temp file immediately (safer)
                    try:
                        if os.path.exists(input_audio_path):
                            os.remove(input_audio_path)
                        try:
                            temp_files.remove(input_audio_path)
                        except ValueError:
                            pass
                    except Exception as e:
                        logging.debug(f"[{request_id}] failed to remove temp file {input_audio_path}: {e}")

    except WebSocketDisconnect:
        logging.info(f"[{request_id}] client disconnected (outside loop)")
    except Exception as e:
        logging.exception(f"[{request_id}] WS main error: {e}")
    finally:
        logging.info(f"[{request_id}] cleaning up {len(temp_files)} files")
        for p in list(temp_files):  # iterate copy to be safe
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                logging.debug(f"[{request_id}] failed to remove leftover temp file {p}")
