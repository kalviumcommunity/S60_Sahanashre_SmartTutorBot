import json
import os

def load_prompt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:   # force utf-8
        return f.read()

def chain_of_thought_prompt(user_query, learner_data):
    base_prompt = load_prompt("prompts/system_prompt_chain_of_thought.txt")
    return (
        base_prompt
        .replace("{user_query}", user_query)
        .replace("{learner_data}", json.dumps(learner_data))
    )

if __name__ == "__main__":
    # Example test case
    user_query = "Can you explain recursion in simple terms?"
    learner_data = {
        "level": "beginner",
        "preferred_style": "analogy",
        "subject": "Computer Science"
    }

    final_prompt = chain_of_thought_prompt(user_query, learner_data)

    # Save outputs
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/chain_of_thought_latest_output_raw.txt", "w") as f:
        f.write(final_prompt)
    with open("evaluation/chain_of_thought_latest_output.json", "w") as f:
        json.dump({"prompt": final_prompt}, f, indent=4)

    print("✅ Chain of Thought prompt generated. Check evaluation/chain_of_thought_latest_output_raw.txt")