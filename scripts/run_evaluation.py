import os, json
from dotenv import load_dotenv
import google.generativeai as genai

# ---------- Setup ----------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def ask_model(prompt):
    resp = model.generate_content(prompt)
    return resp.text.strip()

# ---------- Load dataset ----------
with open("evaluation/dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

judge_template = read("prompts/judge_prompt.txt")

results = []

for sample in dataset:
    user_query = sample["user_query"]
    learner_data = json.dumps(sample["learner_data"], indent=2)
    expected = sample["expected"]

    # Simulated model answer (in real use, call your SmartTutorBot function)
    model_answer = f"(Pretend SmartTutorBot answered) → {expected}"

    # Judge Prompt
    judge_prompt = f"""
{judge_template}

User Query: {user_query}
Learner Data: {learner_data}
Expected Reference: {expected}
Model Output: {model_answer}
"""

    score_text = ask_model(judge_prompt)

    try:
        score_json = json.loads(score_text)
    except Exception:
        score_json = {"error": "Could not parse", "raw": score_text}

    results.append({
        "id": sample["id"],
        "query": user_query,
        "model_answer": model_answer,
        "evaluation": score_json
    })

# ---------- Save results ----------
os.makedirs("evaluation", exist_ok=True)
with open("evaluation/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Evaluation complete → evaluation/results.json")
