import os, json, re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment (.env).")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def main():
    system = read("prompts/system_prompt_smart_tutor.txt")
    user = read("prompts/user_prompt_smart_tutor.txt")

    # Send as two parts: instructions + user input
    resp = model.generate_content([system, user])

    # Raw text the model produced
    try:
        text = resp.candidates[0].content.parts[0].text.strip()
    except Exception:
        text = str(resp)

    print("\n=== RAW MODEL TEXT ===\n")
    print(text)

    # Some models add ```json fences; strip them if present
    text_clean = text.strip()
    if text_clean.startswith("```"):
        text_clean = re.sub(r"^```(\w+)?", "", text_clean).strip()
        text_clean = re.sub(r"```$", "", text_clean).strip()

    # Try to extract JSON object
    match = re.search(r"\{.*\}\s*$", text_clean, re.DOTALL)
    if match:
        text_clean = match.group(0)

    os.makedirs("evaluation", exist_ok=True)

    try:
        data = json.loads(text_clean)
        print("\n Parsed JSON keys:", list(data.keys()))
        with open("evaluation/latest_output.json", "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2, ensure_ascii=False)
        print("Saved → evaluation/latest_output.json")
    except Exception as e:
        print("\n JSON parse failed:", e)
        with open("debug_raw_output.txt", "w", encoding="utf-8") as out:
            out.write(text)
        print("Saved raw → debug_raw_output.txt")

if __name__ == "__main__":
    main()