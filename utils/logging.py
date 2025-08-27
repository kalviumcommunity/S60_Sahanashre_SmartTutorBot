import os

def log_usage(resp, tag="default"):
    """
    Logs token usage from a Gemini response object.
    """
    try:
        usage = resp.usage_metadata
        prompt = usage.prompt_token_count
        cand = usage.candidates_token_count
        total = usage.total_token_count

        print(f"[{tag}] Tokens → prompt: {prompt}, completion: {cand}, total: {total}")

        os.makedirs("logs", exist_ok=True)
        with open("logs/tokens.log", "a", encoding="utf-8") as fh:
            fh.write(f"{tag} | prompt:{prompt} cand:{cand} total:{total}\n")

    except Exception as e:
        print(f"⚠️ Could not log tokens ({tag}):", e)
