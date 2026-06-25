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
    while 1:
        logits: List[float] = model.get_logits_from_input_ids(input_ids[0])
        max_logit_value = max(logits)
        predicted_token_id = logits.index(max_logit_value)
        input_ids[0].append(predicted_token_id)
        predicted_text = model.decode([predicted_token_id])
        print(f"{predicted_text}", end="")


def get_special_tokens() -> None:
    model = Small_LLM_Model()
    tknzr_path = model.get_path_to_tokenizer_file()
    with open(tknzr_path, "r") as file:
        print(file.read())
        print(file)


def get_id_from_char() -> None:
    print(Small_LLM_Model().encode("\\{"))


if __name__ == "__main__":
    get_id_from_char()
