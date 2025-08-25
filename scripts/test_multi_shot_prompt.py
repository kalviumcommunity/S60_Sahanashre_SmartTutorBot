import os
import re
import json
import google.generativeai as genai

# Load API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found. Please set it in your environment.")

genai.configure(api_key=api_key)

def clean_json_fences(text: str) -> str:
    """
    Removes ```json ... ``` fences if present and extracts the JSON block.
    """
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip(), flags=re.DOTALL)
    return text.strip()

def main():
    # Load prompts
    with open("prompts/system_prompt_multi_shot.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    with open("prompts/user_prompt_multi_shot.txt", "r", encoding="utf-8") as f:
        user_prompt = f.read()

    # Run Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content([system_prompt, user_prompt])

    print("\nDEBUG RAW RESPONSE:\n", resp, "\n")

    try:
        content = resp.candidates[0].content.parts[0].text.strip()
        print("\nRAW MODEL CONTENT:\n", content, "\n")

        # 🔧 Clean content
        cleaned = clean_json_fences(content)

        # Parse JSON
        data = json.loads(cleaned)
        print("✅ Parsed JSON Output:\n", json.dumps(data, indent=2))

        # Save outputs
        with open("evaluation/multi_shot_latest_output.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        with open("evaluation/multi_shot_latest_output_raw.txt", "w", encoding="utf-8") as f:
            f.write(content)

    except Exception as e:
        print("⚠️ JSON parse failed:", e)
        with open("evaluation/multi_shot_latest_output_raw.txt", "w", encoding="utf-8") as f:
            f.write(resp.candidates[0].content.parts[0].text if resp.candidates else str(resp))
        print("❌ Could not parse JSON. Check evaluation/multi_shot_latest_output_raw.txt")

if __name__ == "__main__":
    main()