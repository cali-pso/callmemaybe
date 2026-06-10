from typing import List
from llm_sdk import Small_LLM_Model


def main() -> None:
    print("--- Initializing ---")

    model = Small_LLM_Model()

    prompt = "What is the sum of 40 and 2?"

    input_ids = model.encode(prompt)
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()

    predicted_text = ""
    while ("\n" not in predicted_text):
        logits: List[float] = model.get_logits_from_input_ids(input_ids[0])
        max_logit_value = max(logits)
        predicted_token_id = logits.index(max_logit_value)
        predicted_text = model.decode([predicted_token_id])
        print(f"Predicted token: '{predicted_text}'")


if __name__ == "__main__":
    main()
