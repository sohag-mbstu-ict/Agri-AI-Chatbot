import torch
torch.backends.cudnn.benchmark = True # faster kernel selection.
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import psutil
from .preprocess_context import preprocess_context
from utils.gpu_usage_monitoring import print_gpu_usage

class QwenChatbot:
    def __init__(self, model_path="/home/gflml/Chatbot/pretrained_model/Qwen3-4B-Instruct-2507"):
        print("🔄 Loading tokenizer & model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16, #torch.bfloat16,
            device_map="auto",  # Automatically puts model on GPU if available
            trust_remote_code=True)
        # Enable TF32 for RTX 30-series (speeds up matrix multiplications)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model.eval()
        # 🔹 BEGIN GPU CHECK
        if torch.cuda.is_available():
            print("✅ CUDA Available!")
            print("GPU Device Count:", torch.cuda.device_count())
            print("Current Device:", torch.cuda.current_device())
            print("Device Name:", torch.cuda.get_device_name(0))
            print("Model Parameters Device:", next(self.model.parameters()).device)
        else:
            print("⚠️ CUDA not available, using CPU")
        # 🔹 END GPU CHECK

    # --------------------------------------------------
    # 🔹 Detect Query Language (Character Based)
    # --------------------------------------------------
    def detect_query_language(self, text: str) -> str:
        """
        Detect whether a query is primarily English or Bangla
        based on character count.
        """
        english_count = len(re.findall(r"[A-Za-z]", text))
        bangla_count = len(re.findall(r"[\u0980-\u09FF]", text))

        print(f"English letters: {english_count}")
        print(f"Bangla letters : {bangla_count}")

        if english_count > bangla_count:
            return "english"
        else:
            return "bangla"

    # --------------------------------------------------
    # 🔹 RAG Prompt Builder (Bangla/English Adaptive)
    # --------------------------------------------------
    def build_messages(self, query, context):
        # context = preprocess_context(context)
        # print("Inside multi_modal_chatbot/chatbot_app/Qwen3_1_7b.py  --------------%%%preprocess_context%%%%-------------- --\n",context)

        if not context:
            context = "Please ask a clear question so I can help you correctly."

        language = self.detect_query_language(query)

        # ---------------- Bangla Prompt ----------------
        if language == "bangla":
            system_text = (
                "আপনি একজন কঠোর RAG অ্যাসিস্ট্যান্ট। "
                "আপনি শুধুমাত্র প্রদত্ত তথ্য ব্যবহার করে উত্তর দেবেন। "
                "যদি উত্তর স্পষ্টভাবে তথ্যের মধ্যে না থাকে, ঠিক এই বাক্যটি লিখুন: "
                "'প্রদত্ত তথ্যে উত্তর পাওয়া যায়নি।'\n"
                "• শুধুমাত্র প্রদত্ত তথ্য ব্যবহার করুন।\n"
                "• নতুন তথ্য অনুমান করবেন না।\n"
                "• উত্তর স্পষ্ট এবং প্রাঞ্জল করবেন"
            )

            human_text = (
                f"Context:\n{context}\n\n"
                f"Question:\n{query}\n\n"
                "নির্দেশাবলী:\n"
                "1. শুধুমাত্র প্রদত্ত তথ্য ব্যবহার করুন।\n"
                "2. প্রাসঙ্গিক অংশের title বা শিরোনাম উত্তরে লিখবেন না।\n"
                "3. গুরুত্বপূর্ণ শব্দগুলি **বোল্ড** করুন।\n"
                "4. যদি একাধিক পয়েন্ট থাকে, **বুলেট পয়েন্ট** ব্যবহার করুন।\n"
                "5. উত্তর সংক্ষিপ্ত এবং স্পষ্ট রাখুন।\n\n"
                "উত্তর (বাংলায়):"
            )

        # ---------------- English Prompt ----------------
        else:
            system_text = (
                "You are a strict RAG assistant.\n"
                "You MUST answer using ONLY the provided context.\n"
                "Use the titles in the context to locate relevant information, "
                "but DO NOT include titles in your final answer.\n"
                "If the answer is not explicitly stated in the context, reply exactly:\n"
                "'The information is not available in the provided data.'\n"
                "Guidelines:\n"
                "• Prefer information from the context.\n"
                "• Do NOT invent facts or use external knowledge.\n"
                "• Respond clearly and naturally."
            )

            human_text = (
                f"Context:\n{context}\n\n"
                f"Question:\n{query}\n\n"
                "Instructions:\n"
                "1. Answer using the provided context.\n"
                "2. Do NOT include the title of the relevant section in your answer.\n"
                "3. Use **bold text** for important terms.\n"
                "4. If multiple points exist, use bullet points.\n"
                "5. Keep the answer concise and well-structured.\n\n"
                "Answer:"
            )

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": human_text}
        ]

        return messages


    # --------------------------------------------------
    # 🔹 GENERATION (HALLUCINATION-SAFE)
    # --------------------------------------------------
    def generate(self, query, context, max_new_tokens=1512):  # 🔹 Reduce tokens for speed
        # GPU status before
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated Before: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"GPU Memory Reserved Before : {torch.cuda.memory_reserved()/1024**2:.2f} MB")
            print_gpu_usage()

        messages = self.build_messages(query, context)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device, non_blocking=True)

        # Inference (no grad for speed)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # GPU status after
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated After: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"GPU Memory Reserved After : {torch.cuda.memory_reserved()/1024**2:.2f} MB")
            print_gpu_usage()

        # Decode output
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
