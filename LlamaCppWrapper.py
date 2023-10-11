from langchain.llms import LlamaCpp


class LlamaCppWrapper:
    def __init__(self, model_path, n_ctx, top_k, top_p, temperature, repeat_penalty, n_parts, lora_path):
        self.model = LlamaCpp(
            model_path=model_path,
            n_ctx=n_ctx,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            n_parts=n_parts,
            lora_path=lora_path
        )

