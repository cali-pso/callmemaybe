import numpy
import time
import torch
from typing import List, Any
from pydantic import BaseModel, ConfigDict

MAX_TOKENS = 250


class Generator(BaseModel):
    """Generate token batches iteratively using constrained decoding."""

    model: Any
    tokenizer: Any
    constrainer: Any
    eos_token: int = 0
    start_time: float = 0.0
    last_observed_time: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self, model: Any, tokenizer: Any, constrainer: Any, **data: Any
    ) -> None:
        """Initialize the generator with model, tokenizer, and constrainer."""
        super().__init__(
            model=model, tokenizer=tokenizer, constrainer=constrainer, **data
        )
        self.eos_token = self.tokenizer["added_tokens"][0]["id"]
        self.start_time = time.time()
        self.last_observed_time = self.start_time

    def display_generated_texts(
        self, generated_texts: List[str], actives: int
    ) -> None:
        """Print currently generated texts and performance metrics."""
        print("-----------------------------------------------------------\n")
        for text in generated_texts:
            print(text)
        print()
        cur_time = time.time()
        gen_time = cur_time - self.last_observed_time
        full_time = cur_time - self.start_time
        print(
            f"Time since generation start: {full_time:4f}s |",
            f"Batch generation time: {gen_time:4f}s",
            f"| {(actives / gen_time):4f} T/s\n",
        )
        self.last_observed_time = cur_time

    def generate_batch(self, input_ids: List[List[int]]) -> List[str]:
        """Generate token sequences for a batch of input IDs."""
        batch_size = len(input_ids)
        generated_texts = [""] * batch_size
        active_indices = list(range(batch_size))
        tokens_count = 0
        past_key_values = None
        input_tensor = torch.tensor(
            input_ids, device=self.model._device, dtype=torch.long
        )
        attention_mask = (input_tensor != self.eos_token).long()
        self.constrainer.reset_state(batch_size=batch_size)
        while tokens_count <= MAX_TOKENS and len(active_indices) > 0:
            self.display_generated_texts(generated_texts, len(active_indices))
            with torch.no_grad():
                out = self.model._model(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            next_token_logits = out.logits[:, -1, :]
            next_tokens = []
            next_active_indices = []
            rows_to_keep = []
            next_token_logits_np = next_token_logits.cpu().numpy()
            for current_idx, original_idx in enumerate(active_indices):
                if self.constrainer.state[original_idx] != "STATE_FINISHED":
                    seq_logits_np = next_token_logits_np[current_idx]
                    logits_filtered = self.constrainer.apply_constraints(
                        generated_texts[original_idx],
                        seq_logits_np,
                        batch_idx=original_idx,
                    )
                    chosen_token = int(numpy.argmax(logits_filtered))
                else:
                    chosen_token = self.eos_token
                next_tokens.append(chosen_token)
                chosen_trad = (
                    self.constrainer.vocab[chosen_token].replace("Ġ", " ")
                    if chosen_token != self.eos_token
                    else ""
                )
                generated_texts[original_idx] += chosen_trad
                if (
                    chosen_token == self.eos_token
                    or self.constrainer.state[original_idx] == "STATE_FINISHED"
                ):
                    pass
                else:
                    self.constrainer.update_state(
                        chosen_trad,
                        generated_texts[original_idx],
                        batch_idx=original_idx,
                    )
                    next_active_indices.append(original_idx)
                    rows_to_keep.append(current_idx)
            if not next_active_indices:
                break
            input_tensor = input_tensor[rows_to_keep]
            if past_key_values is not None:
                if hasattr(past_key_values, "reorder_cache"):
                    # For modern Hugging Face models using Cache objects
                    keep_indices = torch.tensor(
                        rows_to_keep,
                        device=self.model._device,
                        dtype=torch.long,
                    )
                    past_key_values.reorder_cache(keep_indices)
                else:
                    # Fallback for legacy tuple formats
                    past_key_values = tuple(
                        tuple(
                            tensor[rows_to_keep]
                            if tensor is not None
                            else None
                            for tensor in layer_cache
                        )
                        for layer_cache in past_key_values
                    )
            attention_mask = attention_mask[rows_to_keep]
            new_mask_col = torch.ones(
                (len(rows_to_keep), 1),
                device=self.model._device,
                dtype=torch.long,
            )
            attention_mask = torch.cat([attention_mask, new_mask_col], dim=1)
            active_next_tokens = [next_tokens[i] for i in rows_to_keep]
            input_tensor = torch.tensor(
                active_next_tokens, device=self.model._device, dtype=torch.long
            ).unsqueeze(1)
            active_indices = next_active_indices
            tokens_count += 1
        return generated_texts
