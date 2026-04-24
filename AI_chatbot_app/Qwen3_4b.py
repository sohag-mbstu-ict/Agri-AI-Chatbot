import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_path="/home/gflml/Chatbot/pretrained_model/Qwen3-4B-Instruct-2507"):
        #/home/gflml/Chatbot/pretrained_model/Qwen3-4B-Instruct-2507
        print("🔄 Loading tokenizer & model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True)

        self.model.eval()
        print("✅ Model loaded successfully!")

    # --------------------------------------------------
    # 🔹 QWEN-COMPATIBLE RAG PROMPT
    # --------------------------------------------------
    def build_messages(self, query, context):
        if not context:
            context = "Please ask a clear question so I can help you correctly."

        messages = [
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
        return messages


    # --------------------------------------------------
    # 🔹 GENERATION (HALLUCINATION-SAFE)
    # --------------------------------------------------
    def generate(self, query, context, max_new_tokens=1500):
        # print("0000000000000000000000000000000000000000000000 context : ",context)
        messages = self.build_messages(query, context)
        # 🔹 Apply Qwen chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # ✅ deterministic
                temperature=None,         # ✅ remove conflicts
                top_p=None,
                eos_token_id=self.tokenizer.eos_token_id)
        # Remove prompt tokens
        generated = outputs[0][inputs.input_ids.shape[-1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer
    
    def build_messages_for_fallback(self, query):
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful, intelligent, and friendly AI assistant.\n"
                    "Provide clear and accurate answers.\n"
                    "Interpret spelling mistakes correctly.\n"
                    "Be professional and conversational.\n"
                )
            },
            {
                "role": "user",
                "content": query
            }
        ]

    def generate_answer_for_fallback(self, query, max_new_tokens=512):
        messages = self.build_messages_for_fallback(query)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id)
        generated = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def start_chat(self, rag_context_lookup=None):
        """
        rag_context_lookup: optional function that returns context string given a user query
        """
        print("\n💬 Qwen RAG Chatbot is ready! Type 'exit' to quit.\n")
        while True:
            user_input = input("Ask your question: ")
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Exiting chat. Bye!")
                break

            # Retrieve context from RAG dataset if function is provided
            context = None
            if rag_context_lookup:
                context = rag_context_lookup(user_input)

            answer = self.generate(user_input, context=context)
            print("\n🤖 Answer:", answer)
            print("-" * 60)
