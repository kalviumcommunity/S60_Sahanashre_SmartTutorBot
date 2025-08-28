import os, json, re
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

# ---------- Setup ----------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

REQ_KEYS = {
    "topic","grade_level","learning_objective","lesson_summary","key_points",
    "worked_examples","practice_problems","assessment_quiz",
    "follow_up_recommendations","references","estimated_time_minutes"
}

# ---------- Example Function ----------
def get_weather(city: str, unit: str = "C"):
    """Dummy weather lookup"""
    return {
        "city": city,
        "unit": unit,
        "forecast": "Sunny",
        "temperature": 28 if unit == "C" else 82
    }

# ---------- Helpers ----------
def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def extract_text(resp):
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
    except:
        pass
    try:
        return resp.candidates[0].content.parts[0].text.strip()
    except:
        return str(resp)

def clean_to_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t).strip()
        t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}\s*$", t, re.DOTALL)
    return m.group(0) if m else t

# ---------- Main ----------
def main():
    system = read("prompts/system_prompt_one_shot.txt")
    user   = read("prompts/user_prompt_one_shot.txt")

    prompt = f"{system}\n\n# New Task\n{user}"

    # ✅ Correct function schema
    tools = [
        Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="get_weather",
                    description="Get the weather forecast for a given city",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["C", "F"],
                                "description": "Temperature unit"
                            }
                        },
                        "required": ["city"]
                    }
                )
            ]
        )
    ]

    resp = model.generate_content(
        prompt,
        tools=tools,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            top_p=0.9,
            top_k=40
        )
    )

    print("\n=== RAW MODEL RESPONSE ===\n")
    print(resp)

    # Check if model requested function call
    fn_call = None
    try:
        fn_call = resp.candidates[0].content.parts[0].function_call
    except:
        pass

    if fn_call:
        fn_name = fn_call.name
        args = json.loads(fn_call.args)
        print(f"\n📞 Model requested function call: {fn_name} with args {args}")

        if fn_name == "get_weather":
            result = get_weather(**args)
            print("\n✅ Function executed. Result:", result)

            # Send result back to model for final response
            followup = model.generate_content(
                [{"role": "model", "parts": [result]}]
            )
            print("\n=== FINAL MODEL RESPONSE ===\n")
            print(followup.text)
            return

    # Fallback → structured JSON response
    raw = extract_text(resp)
    print("\n=== RAW MODEL TEXT (one-shot) ===\n")
    print(raw)

    try:
        text = clean_to_json(raw)
        data = json.loads(text)

        for k in REQ_KEYS:
            if k not in data:
                data[k] = None

        os.makedirs("evaluation", exist_ok=True)
        with open("evaluation/one_shot_latest_output.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print("\n✅ Parsed JSON successfully.")
        print("Saved → evaluation/one_shot_latest_output.json")
    except Exception as e:
        print("\n⚠️ JSON parse failed:", e)
        os.makedirs("evaluation", exist_ok=True)
        with open("evaluation/one_shot_latest_output_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        print("Saved raw → evaluation/one_shot_latest_output_raw.txt")


if __name__ == "__main__":
    main()
