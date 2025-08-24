import os, json, re
from dotenv import load_dotenv
import google.generativeai as genai

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

def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def extract_text(resp):
    # Be robust to different SDK return shapes
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
    except Exception:
        pass
    try:
        return resp.candidates[0].content.parts[0].text.strip()
    except Exception:
        return str(resp)

def clean_to_json(text: str) -> str:
    # Remove markdown fences if any and extract the last JSON object
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(\w+)?", "", t).strip()
        t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}\s*$", t, re.DOTALL)
    return m.group(0) if m else t

def log_usage(resp, tag="one-shot"):
    try:
        usage = resp.usage_metadata
        prompt = usage.prompt_token_count
        cand = usage.candidates_token_count
        total = usage.total_token_count
        os.makedirs("logs", exist_ok=True)
        with open("logs/tokens.log", "a", encoding="utf-8") as fh:
            fh.write(f"{tag} | prompt:{prompt} cand:{cand} total:{total}\n")
        print(f"Tokens → prompt:{prompt} cand:{cand} total:{total}")
    except Exception:
        pass

def main():
    system = read("prompts/system_prompt_one_shot.txt")
    user   = read("prompts/user_prompt_one_shot.txt")

    # Concatenate instructions + task (Gemini doesn't need explicit roles)
    prompt = f"{system}\n\n# New Task\n{user}"
    resp = model.generate_content(prompt)

    log_usage(resp, tag="one-shot")

    raw = extract_text(resp)
    print("\n=== RAW MODEL TEXT (one-shot) ===\n")
    print(raw)

    text = clean_to_json(raw)

    os.makedirs("evaluation", exist_ok=True)

    try:
        data = json.loads(text)
        missing = sorted(list(REQ_KEYS - set(data.keys())))
        extra   = sorted(list(set(data.keys()) - REQ_KEYS))
        print("\n✅ Parsed JSON. Missing keys:", missing, " Extra keys:", extra)
        with open("evaluation/one_shot_latest_output.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("Saved → evaluation/one_shot_latest_output.json")
    except Exception as e:
        print("\n⚠️ JSON parse failed:", e)
        with open("evaluation/one_shot_latest_output_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        print("Saved raw → evaluation/one_shot_latest_output_raw.txt")

if __name__ == "__main__":
    main()