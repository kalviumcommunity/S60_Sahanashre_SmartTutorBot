import json
import os

def load_prompt(filepath):
    with open(filepath, "r") as f:
        return f.read()

def dynamic_prompt(user_query, learner_data):
    system_prompt = load_prompt("prompts/system_prompt_dynamic.txt")
    user_prompt = load_prompt("prompts/user_prompt_dynamic.txt")

    combined_prompt = (
        system_prompt.replace("{user_query}", user_query).replace("{learner_data}", json.dumps(learner_data, indent=2))
        + "\n\n"
        + user_prompt.replace("{user_query}", user_query).replace("{learner_data}", json.dumps(learner_data, indent=2))
    )
    return combined_prompt

if __name__ == "__main__":
    # Example input
    user_query = "Can you explain recursion in simple terms?"
    learner_data = {
        "level": "beginner",
        "preferred_style": "analogy",
        "subject": "Computer Science"
    }

    final_prompt = dynamic_prompt(user_query, learner_data)

    # Save full prompt for debugging
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/dynamic_latest_output_raw.txt", "w") as f:
        f.write(final_prompt)
    with open("evaluation/dynamic_latest_output.json", "w") as f:
        json.dump({"prompt": final_prompt}, f, indent=4)

    # ✅ Instead of printing the prompt, print input as JSON
    output_json = {
        "user_query": user_query,
        "learner_data": learner_data
    }
    print(json.dumps(output_json, indent=4))
