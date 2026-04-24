import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenChatbot:
    """
    Gemma-3-1B-IT RAG-safe inference chatbot
    """

    def __init__(
        self,
        model_path="/home/gflml/Chatbot/pretrained_model/gemma/gemma-3-1b-it",
        dtype=torch.bfloat16
    ):
        print("🔄 Loading Gemma tokenizer & model...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto"
        )

        self.model.eval()
        print("✅ Gemma model loaded successfully!")

    # --------------------------------------------------
    # 🔹 GEMMA-NATIVE RAG PROMPT
    # --------------------------------------------------
    def build_prompt(self, query: str, context: str) -> str:
        """
        Builds a strict RAG instruction prompt for Gemma
        """

        if not context or not context.strip():
            context = "The information is not available in the provided data."

        prompt = (
            "You are a strict RAG assistant.\n\n"
            "You MUST answer using ONLY the provided context.\n"
            "If the answer is not explicitly stated in the context, reply EXACTLY with:\n"
            "The information is not available in the provided data.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

        return prompt

    # --------------------------------------------------
    # 🔹 GENERATION (DETERMINISTIC & HALLUCINATION-SAFE)
    # --------------------------------------------------
    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 2500
    ) -> str:
        """
        Generates an answer strictly grounded in context
        """
        prompt = self.build_prompt(query, context)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,                 # deterministic
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Remove prompt tokens from output
        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return answer

    # --------------------------------------------------
    # 🔹 INTERACTIVE CHAT (OPTIONAL)
    # --------------------------------------------------
    def start_chat(self, context):
        print("\n💬 Gemma Chatbot is ready! Type 'exit' to quit.\n")

        while True:
            query = input("Ask your question: ").strip()

            if query.lower() in {"exit", "quit"}:
                print("👋 Exiting chat. Bye!")
                break

            answer = self.generate(query, context)
            print("\n🤖 Answer:")
            print(answer)
            print("-" * 60)




# # --------------------------------------------------
# # 🔹 RUN
# # --------------------------------------------------
# if __name__ == "__main__":
#     chatbot = GemmaRAGChatbot()
#     context = "A rooftop garden is a green space cultivated on a building's roof, offering aesthetic beauty, food production (rooftop farming), wildlife habitats, and environmental benefits like urban heat reduction and improved air quality, using systems from simple containers to complex green roofs with plants, soil, and water management. "
#     chatbot.start_chat(context)


