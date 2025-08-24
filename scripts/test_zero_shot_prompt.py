import os, json, re
from dotenv import load_dotenv
import google.generativeai as genai

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

def read(p): 
    with open(p, "r", encoding="utf-8") as f: 
        return f.read().strip()

def clean_to_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(\w+)?", "", t).strip()
        t = re.sub(r"```$", "", t).strip()
    # grab the last JSON object if extra text appears
    m = re.search(r"\{.*\}\s*$", t, re.DOTALL)
    return m.group(0) if m else t

def main():
    system = read("prompts/system_prompt_zero_shot.txt")
    user   = read("prompts/user_prompt_zero_shot.txt")

    resp = model.generate_content([system, user])

    # Extract raw text
    try:
        raw = resp.candidates[0].content.parts[0].text.strip()
    except Exception:
        raw = str(resp)

    print("\n=== RAW MODEL TEXT (zero-shot) ===\n")
    print(raw)

    text = clean_to_json(raw)
    os.makedirs("evaluation", exist_ok=True)

    try:
        data = json.loads(text)
        missing = sorted(list(REQ_KEYS - set(data.keys())))
        extra   = sorted(list(set(data.keys()) - REQ_KEYS))
        print("\n✅ Parsed JSON. Missing keys:", missing, " Extra keys:", extra)
        out_path = "evaluation/zero_shot_latest_output.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved → {out_path}")
    except Exception as e:
        print("\n⚠️ JSON parse failed:", e)
        with open("evaluation/zero_shot_latest_output_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        print("Saved raw → evaluation/zero_shot_latest_output_raw.txt")

if __name__ == "__main__":
    main()