
from langchain_core.prompts import ChatPromptTemplate
def is_bangla(text: str) -> bool:
    return ""






# old2
def build_rag_prompt_for_groq_qween_32B(context, query):
    """
    Build a RAG prompt for Groq Qwen-32B.
    Automatically outputs in Bangla if the query contains Bangla,
    otherwise in English.
    Optimized for clear answers, bullet points, and bold keywords.
    Titles are used in context for retrieval, but MUST NOT appear in the answer.
    """
    if is_bangla(query):
        system_text = (
            "আপনি একজন দক্ষ এবং সহায়ক RAG অ্যাসিস্ট্যান্ট, "
            "বিশেষজ্ঞ শহুরে কৃষি এবং ছাদবাগান। "
            "আপনি শুধুমাত্র প্রদত্ত তথ্য থেকে উত্তর দেবেন। "
            "প্রদত্ত তথ্য থেকে কোন শিরোনাম ব্যবহার করবেন, কিন্তু উত্তরে শিরোনাম লিখবেন না। "
            "যদি তথ্য পাওয়া না যায়, সঠিকভাবে লিখুন: 'প্রদত্ত তথ্যে উত্তর পাওয়া যায়নি।'\n"
            "• শুধুমাত্র প্রদত্ত তথ্য ব্যবহার করুন।\n"
            "• নতুন তথ্য অনুমান করবেন না।\n"
            "• উত্তর স্পষ্ট এবং প্রাঞ্জল হোক।"
        )

        human_text = (
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "নির্দেশাবলী:\n"
            "1. শুধুমাত্র প্রদত্ত তথ্য ব্যবহার করুন।\n"
            "2. প্রাসঙ্গিক অংশের শিরোনাম **উত্তরে লিখবেন না।**\n"
            "3. গুরুত্বপূর্ণ শব্দগুলি **বোল্ড** করুন।\n"
            "4. যদি একাধিক পয়েন্ট থাকে, **বুলেট পয়েন্ট** ব্যবহার করুন।\n"
            "5. উত্তর সংক্ষিপ্ত এবং স্পষ্ট রাখুন।\n\n"
            "উত্তর (বাংলায়):"
        )
    else:
        system_text = (
            "You are a knowledgeable and helpful RAG assistant specialized in Urban Agriculture, "
            "Rooftop Gardening, and general factual knowledge. "
            "Use the titles in the context to find relevant info, but do NOT include titles in your answer. "
            "If the answer is not found, reply exactly: "
            "'The information is not available in the provided data.'\n"
            "• Prefer information from the context.\n"
            "• Do NOT invent facts or use external knowledge.\n"
            "• Respond clearly and naturally."
        )

        human_text = (
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Instructions:\n"
            "1. Answer using the provided context.\n"
            "2. Do NOT include the **title** of the relevant section in your answer.\n"
            "3. Use **bold text** for important terms or names.\n"
            "4. If the answer contains multiple points, present them as **bullet points**.\n"
            "5. Keep the answer concise, clear, and well-structured.\n\n"
            "Answer:"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", human_text)
    ])
    return prompt


# old 3
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
                    "DO NOT explain your reasoning.\n"
                    "DO NOT include analysis, thoughts, or commentary.\n\n"

                    "MATCHING RULE (CRITICAL):\n"
                    "- If the user question semantically matches a **Title** in the context,\n"
                    "  return the corresponding **Content** verbatim.\n"
                    "- Question marks, capitalization, or minor wording differences\n"
                    "  MUST NOT prevent matching.\n\n"

                    "SEMANTIC MATCHING RULE (CRITICAL):\n"
                    "- If the user question contains spelling mistakes,\n"
                    "  partial words, or minor wording differences,\n"
                    "  you MUST still match it semantically with the Titles or Content.\n"
                    "- Example: 'banan' should match 'banana'.\n"
                    "- Example: 'mngo disease' should match 'mango disease'.\n\n"

                    "FORMATTING RULES (MANDATORY):\n"
                    "- Use '\\n' for every new line\n"
                    "- Use '-' ONLY at the beginning of a line for bullet points\n"
                    "- Each bullet point MUST be on a separate line\n"
                    "- DO NOT place '-' in the middle of a sentence\n"
                    "- Use '**word**' ONLY for emphasis if the same word exists in the context\n"
                    "- DO NOT invent, merge, reorder, or rephrase content\n\n"

                    "FAILURE RULE:\n"
                    "If the answer is not explicitly stated in the context,\n"
                    "reply EXACTLY with:\n"
                    "The information is not available in the provided data. Please ask a clear question so I can help you correctly.\n\n"

                    "LIST RULE:\n"
                    "- DO NOT summarize or shorten lists\n"
                    "- Extract information verbatim from the context"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{query}\n\n"
                    "ANSWER RULES:\n"
                    "- Use ONLY facts from the context\n"
                    "- Preserve original wording and emojis\n"
                    "- If multiple matching Titles exist, return the most relevant Content\n"
                    "- If listing items, start each item with '- ' on a new line\n"
                )
            }
        ]
        return messages


