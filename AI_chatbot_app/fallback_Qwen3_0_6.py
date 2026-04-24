import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenChatbot:
    def __init__(self, model_path="/home/gflml/Chatbot/pretrained_model/Qwen3-0.6B"):
        print("🔄 Loading tokenizer & model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # 🔥 CRITICAL FIX FOR Qwen3-0.6B (broken generation_config)
        self.model.generation_config.max_new_tokens = 512   # must be INT
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None

        self.model.eval()
        print("✅ Model loaded successfully!")

    # --------------------------------------------------
    # 🔹 QWEN-COMPATIBLE RAG PROMPT
    # --------------------------------------------------
    def build_messages(self, query, context):
        if not context:
            context = "The information is not available in the provided data."

        return [
            {
                "role": "system",
                "content": (
                    "You are a strict RAG assistant.\n"
                    "You MUST answer using ONLY the provided context.\n"
                    "DO NOT use prior knowledge.\n"
                    "DO NOT guess or assume.\n"
                    "If the answer is not explicitly stated in the context, reply exactly:\n"
                    "'The information is not available in the provided data.'"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{query}\n\n"
                    "Answer rules:\n"
                    "- Use ONLY facts from context\n"
                    "- Be concise\n"
                    "- Use bullet points if applicable\n"
                )
            }
        ]

    # --------------------------------------------------
    # 🔹 GENERATION (HALLUCINATION-SAFE)
    # --------------------------------------------------
    def generate(self, query, context, max_new_tokens=150):
        # 🔒 Hard safety cast
        max_new_tokens = int(max_new_tokens)

        messages = self.build_messages(query, context)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy → deterministic
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Remove prompt tokens
        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return answer

    # --------------------------------------------------
    # 🔹 CLI CHAT LOOP
    # --------------------------------------------------
    def start_chat(self, rag_context_lookup=None):
        """
        rag_context_lookup: optional function(query) -> context string
        """
        print("\n💬 Qwen RAG Chatbot is ready! Type 'exit' to quit.\n")

        while True:
            user_input = input("Ask your question: ").strip()

            if user_input.lower() in {"exit", "quit"}:
                print("👋 Exiting chat. Bye!")
                break

            context = None
            if rag_context_lookup:
                context = rag_context_lookup(user_input)

            answer = self.generate(user_input, context=context)

            print("\n🤖 Answer:")
            print(answer)
            print("-" * 60)
