import os
import json
import base64
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8001"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-3.8-flash")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/interactions"
)


# =========================================================
# SHIVI PERSONAL AI
# =========================================================

SHIVI_SYSTEM_PROMPT = """
You are SHIVI, a personal AI assistant.

You are NOT a real-estate-only assistant.

Your job is to help the user with general personal and digital tasks.

You can help with:
- general questions
- Hindi and Hinglish conversation
- personal productivity
- study assistance
- coding and programming
- Android development
- project development
- web research
- planning
- calculations
- writing
- technology
- computer and phone assistance
- AI and automation
- everyday tasks

You are the user's general-purpose personal AI assistant.

Do NOT introduce yourself as a Real Estate AI.

Do NOT turn normal questions into real-estate questions.

If the user asks about real estate, you may answer normally,
but remain SHIVI Personal AI unless the request comes through
the EstateAI assistant.

Be helpful, concise and professional.

If the user speaks Hindi or Hinglish,
respond in Hindi/Hinglish.

Never claim that you performed an action if you did not actually perform it.
"""


# =========================================================
# ESTATE AI
# =========================================================

REAL_ESTATE_SYSTEM_PROMPT = """
You are EstateAI, a professional real-estate AI assistant.

You are NOT the user's general personal assistant.

Your job is specifically to help with real estate.

You can help with:
- property search
- property requirements
- property comparison
- property budgets
- BHK requirements
- locations
- buying property
- renting property
- selling property
- property questions
- lead qualification
- customer requirements
- connecting customers with property agents

You may discuss the property data supplied in the user's message.

Never invent a real property listing unless property data is provided.

If demo property data is provided, clearly use it as demo/property
information supplied by the application.

Be concise and professional.

If the user speaks Hindi or Hinglish,
respond in Hindi/Hinglish.

Do NOT behave like the user's general personal AI.

Do NOT introduce yourself as the user's personal assistant.

Your identity is EstateAI.
"""


# =========================================================
# SELECT AI PERSONALITY
# =========================================================

def get_system_prompt(assistant_type):

    assistant_type = str(
        assistant_type or "shivi"
    ).strip().lower()

    if assistant_type in (
        "real_estate",
        "real-estate",
        "estateai",
        "estate"
    ):
        return REAL_ESTATE_SYSTEM_PROMPT

    return SHIVI_SYSTEM_PROMPT


# =========================================================
# EXTRACT GEMINI RESPONSE
# =========================================================

def extract_text_from_result(result):

    # Direct output_text
    output_text = result.get("output_text")

    if isinstance(output_text, str):

        output_text = output_text.strip()

        if output_text:
            return output_text


    # Interactions API:
    # steps -> model_output -> content -> text

    steps = result.get("steps", [])

    if isinstance(steps, list):

        text_parts = []

        for step in steps:

            if not isinstance(step, dict):
                continue

            if step.get("type") != "model_output":
                continue

            content = step.get("content", [])

            if not isinstance(content, list):
                continue

            for part in content:

                if not isinstance(part, dict):
                    continue

                if part.get("type") != "text":
                    continue

                text = part.get("text")

                if (
                    isinstance(text, str)
                    and text.strip()
                ):
                    text_parts.append(
                        text.strip()
                    )

        if text_parts:
            return "\n".join(
                text_parts
            ).strip()


    # Generic outputs fallback

    outputs = result.get("outputs", [])

    if isinstance(outputs, list):

        text_parts = []

        for item in outputs:

            if not isinstance(item, dict):
                continue

            text = item.get("text")

            if (
                isinstance(text, str)
                and text.strip()
            ):
                text_parts.append(
                    text.strip()
                )

            content = item.get("content", [])

            if isinstance(content, list):

                for part in content:

                    if not isinstance(part, dict):
                        continue

                    text = part.get("text")

                    if (
                        isinstance(text, str)
                        and text.strip()
                    ):
                        text_parts.append(
                            text.strip()
                        )

        if text_parts:
            return "\n".join(
                text_parts
            ).strip()


    # Legacy candidates fallback

    candidates = result.get("candidates", [])

    if isinstance(candidates, list):

        text_parts = []

        for candidate in candidates:

            if not isinstance(candidate, dict):
                continue

            content = candidate.get(
                "content",
                {}
            )

            if not isinstance(content, dict):
                continue

            parts = content.get(
                "parts",
                []
            )

            if not isinstance(parts, list):
                continue

            for part in parts:

                if not isinstance(part, dict):
                    continue

                text = part.get("text")

                if (
                    isinstance(text, str)
                    and text.strip()
                ):
                    text_parts.append(
                        text.strip()
                    )

        if text_parts:
            return "\n".join(
                text_parts
            ).strip()


    return None


# =========================================================
# ASK GEMINI
# =========================================================

def ask_gemini(message, assistant_type):

    if not GEMINI_API_KEY:

        return (
            "Gemini API key is not configured."
        )


    system_prompt = get_system_prompt(
        assistant_type
    )


    payload = {
        "model": GEMINI_MODEL,

        "input": message,

        "system_instruction": system_prompt,

        "generation_config": {
            "thinking_level": "low"
        }
    }


    data = json.dumps(
        payload,
        ensure_ascii=False
    ).encode("utf-8")


    request = Request(
        GEMINI_URL,

        data=data,

        headers={
            "Content-Type":
                "application/json",

            "x-goog-api-key":
                GEMINI_API_KEY
        },

        method="POST"
    )


    try:

        print(
            "[SHIVI AI] Assistant:",
            assistant_type
        )

        print(
            "[SHIVI AI] Sending request to Gemini..."
        )


        with urlopen(
            request,
            timeout=60
        ) as response:

            raw = response.read().decode(
                "utf-8"
            )


        result = json.loads(raw)


        print(
            "[GEMINI RAW RESPONSE]"
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            )
        )


        answer = extract_text_from_result(
            result
        )


        if answer:

            print(
                "[SHIVI AI] Response received."
            )

            return answer


        print(
            "[SHIVI AI] No text found."
        )


        return (
            "Gemini returned a response, "
            "but no text was found."
        )


    except HTTPError as e:

        try:

            error_body = (
                e.read()
                .decode("utf-8")
            )

        except Exception:

            error_body = ""


        print(
            "[GEMINI HTTP ERROR]",
            e.code
        )

        print(
            error_body
        )


        return (
            "Gemini API error "
            + str(e.code)
            + ": "
            + error_body[:1000]
        )


    except URLError as e:

        print(
            "[GEMINI NETWORK ERROR]",
            repr(e)
        )


        return (
            "Gemini network error: "
            + str(e.reason)
        )


    except TimeoutError:

        print(
            "[GEMINI TIMEOUT]"
        )


        return (
            "Gemini request timed out."
        )


    except Exception as e:

        print(
            "[GEMINI ERROR]",
            repr(e)
        )


        return (
            "Gemini error: "
            + str(e)
        )



# =========================================================
# GEMINI VISION
# =========================================================

def ask_gemini_vision(prompt, image_base64, mime_type):
    """Analyze a base64 image using the Gemini Interactions API."""
    if not GEMINI_API_KEY:
        return "Gemini API key is not configured."

    if not isinstance(image_base64, str) or not image_base64.strip():
        return "image_base64 is required."

    # Accept a data URL such as data:image/png;base64,AAAA...
    image_base64 = image_base64.strip()
    if image_base64.startswith("data:"):
        try:
            header, image_base64 = image_base64.split(",", 1)
            match = re.match(r"data:([^;]+);base64", header)
            if match:
                mime_type = match.group(1)
        except ValueError:
            return "Invalid image data URL."

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except Exception:
        return "Image must be valid base64 data."

    # Keep request size bounded for a small/free web service.
    if len(image_bytes) > 12 * 1024 * 1024:
        return "Image is too large. Please choose an image smaller than 12 MB."

    allowed_mime_types = {
        "image/jpeg", "image/png", "image/webp", "image/gif"
    }
    mime_type = str(mime_type or "image/jpeg").lower().strip()
    if mime_type not in allowed_mime_types:
        return "Unsupported image type. Use JPEG, PNG, WEBP, or GIF."

    prompt = str(prompt or "").strip()
    if not prompt:
        prompt = (
            "Is image ko dhyan se dekho aur Hindi mein samjhao. "
            "Agar screenshot hai to visible text aur error bhi batao."
        )

    payload = {
        "model": GEMINI_VISION_MODEL,
        "input": [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "data": image_base64,
                "mime_type": mime_type
            }
        ],
        "system_instruction": (
            "You are SHIVI, a helpful personal AI assistant. "
            "Describe the provided image accurately. If the user writes "
            "in Hindi or Hinglish, respond in Hindi or Hinglish. "
            "For screenshots, read visible text and explain visible errors. "
            "Do not claim to see details that are not visible."
        )
    }

    request = Request(
        GEMINI_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        },
        method="POST"
    )

    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))

        answer = extract_text_from_result(result)
        if answer:
            return answer
        return "Gemini Vision returned a response, but no text was found."

    except HTTPError as e:
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            error_body = ""
        print("[GEMINI VISION HTTP ERROR]", e.code, error_body[:2000])
        return "Gemini Vision API error " + str(e.code) + ": " + error_body[:700]

    except URLError as e:
        print("[GEMINI VISION NETWORK ERROR]", repr(e))
        return "Gemini Vision network error: " + str(e.reason)

    except TimeoutError:
        return "Gemini Vision request timed out."

    except Exception as e:
        print("[GEMINI VISION ERROR]", repr(e))
        return "Gemini Vision error: " + str(e)


# =========================================================
# HTTP SERVER
# =========================================================

class SHIVIProductionHandler(
    BaseHTTPRequestHandler
):


    def send_json(
        self,
        payload,
        status=200
    ):

        body = json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8")


        self.send_response(status)


        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )


        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )


        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )


        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )


        self.end_headers()


        self.wfile.write(body)


    def do_OPTIONS(self):

        self.send_json(
            {},
            204
        )


    def do_GET(self):

        if self.path == "/":

            self.send_json(
                {
                    "ok": True,
                    "name": "SHIVI Production AI",
                    "status": "online",
                    "model": GEMINI_MODEL
                }
            )

            return


        if self.path == "/health":

            self.send_json(
                {
                    "ok": True,
                    "name": "SHIVI Production AI",
                    "status": "online",
                    "model": GEMINI_MODEL,
                    "ai_configured":
                        bool(GEMINI_API_KEY)
                }
            )

            return


        self.send_json(
            {
                "ok": False,
                "error": "Not found"
            },
            404
        )


    def do_POST(self):

        if self.path == "/vision":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 18 * 1024 * 1024:
                    self.send_json(
                        {"ok": False, "error": "Invalid request size."},
                        413 if length > 18 * 1024 * 1024 else 400
                    )
                    return

                raw = self.rfile.read(length)
                data = json.loads(raw.decode("utf-8"))

                image_base64 = (
                    data.get("image_base64")
                    or data.get("imageBase64")
                    or data.get("image")
                    or data.get("base64")
                    or ""
                )
                prompt = data.get("prompt") or data.get("message") or ""
                mime_type = data.get("mime_type") or data.get("mimeType") or "image/jpeg"

                answer = ask_gemini_vision(prompt, image_base64, mime_type)
                ok = not answer.startswith((
                    "Gemini API key is not configured.",
                    "image_base64 is required.",
                    "Image must be valid base64 data.",
                    "Unsupported image type.",
                    "Image is too large.",
                    "Invalid image data URL.",
                    "Gemini Vision API error",
                    "Gemini Vision network error",
                    "Gemini Vision request timed out.",
                    "Gemini Vision error:",
                    "Gemini Vision returned a response, but no text was found."
                ))
                self.send_json(
                    {
                        "ok": ok,
                        "answer": answer,
                        "model": GEMINI_VISION_MODEL
                    },
                    200 if ok else 502
                )
            except Exception as e:
                print("[SHIVI VISION REQUEST ERROR]", repr(e))
                self.send_json(
                    {"ok": False, "error": "Invalid vision request: " + str(e)},
                    400
                )
            return

        if self.path != "/chat":
            self.send_json(
                {"ok": False, "error": "Not found"},
                404
            )
            return

        try:

            length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )


            raw = self.rfile.read(
                length
            )


            data = json.loads(
                raw.decode("utf-8")
            )


            # -------------------------------------------------
            # MESSAGE
            # -------------------------------------------------

            message = str(
                data.get(
                    "message",
                    ""
                )
            ).strip()


            if not message:

                self.send_json(
                    {
                        "ok": False,
                        "error":
                            "message is required"
                    },
                    400
                )

                return


            # -------------------------------------------------
            # ASSISTANT TYPE
            # -------------------------------------------------

            assistant_type = str(
                data.get(
                    "assistant",
                    "shivi"
                )
            ).strip().lower()


            # Only allow our two official assistants

            if assistant_type not in (
                "shivi",
                "real_estate"
            ):

                assistant_type = "shivi"


            print(
                "[SHIVI CLOUD] Assistant:",
                assistant_type
            )


            print(
                "[SHIVI CLOUD] Message:",
                message
            )


            # -------------------------------------------------
            # ASK GEMINI
            # -------------------------------------------------

            answer = ask_gemini(
                message,
                assistant_type
            )


            # -------------------------------------------------
            # RESPONSE
            # -------------------------------------------------

            self.send_json(
                {
                    "ok": True,

                    "assistant":
                        assistant_type,

                    "answer":
                        answer,

                    "model":
                        GEMINI_MODEL
                }
            )


        except Exception as e:

            print(
                "[SHIVI CLOUD ERROR]",
                repr(e)
            )


            self.send_json(
                {
                    "ok": False,
                    "error":
                        "API error: "
                        + str(e)
                },
                500
            )


    def log_message(
        self,
        format,
        *args
    ):

        print(
            "[SHIVI CLOUD]",
            format % args
        )


# =========================================================
# START SERVER
# =========================================================

def main():

    server = ThreadingHTTPServer(
        (HOST, PORT),
        SHIVIProductionHandler
    )


    print(
        "==================================="
    )

    print(
        "SHIVI PRODUCTION AI"
    )

    print(
        "==================================="
    )

    print(
        "Host :",
        HOST
    )

    print(
        "Port :",
        PORT
    )

    print(
        "Model:",
        GEMINI_MODEL
    )

    print(
        "API  : Interactions API"
    )

    print(
        "Assistants:"
    )

    print(
        "  - shivi"
    )

    print(
        "  - real_estate"
    )

    print(
        "Health: /health"
    )

    print(
        "Chat  : /chat"
    )

    print()


    print(
        "Production AI server is running."
    )


    try:

        server.serve_forever()


    except KeyboardInterrupt:

        print(
            "SHIVI Production AI stopped."
        )


    finally:

        server.server_close()


if __name__ == "__main__":

    main()