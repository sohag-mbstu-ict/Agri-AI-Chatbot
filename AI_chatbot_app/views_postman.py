from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import StreamingHttpResponse
import re
import json
import time
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from .query_recommendation import recommend_titles
from .word_autocomplete import words_suggest
from .query_answer import get_answer_using_RAG, split_query_score
from .query_image import needs_images, extract_crop_name, get_disease_images_from_query

class Word_AutocompleteAPIView(APIView):
    def get(self, request):
        q = request.GET.get("q", "").strip()
        if not q:
            return Response({"suggested_words": []}, status=status.HTTP_200_OK)
        # Get suggestions
        suggestions = words_suggest(q, top_k=5)

        # Format for production-ready JSON
        response_data = {
            "Input_Character": q,
            "suggested_words": [
                {"word": word} for word in suggestions
            ]}

        print("suggestions : ------------------ ", suggestions)
        return Response(response_data, status=status.HTTP_200_OK)



class Query_Suggestions_APIView(APIView):
    def post(self, request):
        selected_word = request.data.get("selected")
        print("selected_word ------------------------ : ", selected_word)
        if not selected_word or not isinstance(selected_word, str):
            return Response(
                {"queries": []},
                status=status.HTTP_200_OK)
        selected_word = selected_word.strip()
        query_with_score = recommend_titles(selected_word)  # returns ["query : score", ...]
        # Build production-ready JSON
        response_queries = []
        for item in query_with_score:
            if ":" in item:
                query_text, score = map(str.strip, item.rsplit(":", 1))
                score = float(score)
            else:
                query_text = item
                score = None

            response_queries.append({
                "selected_word":selected_word,
                "suggested_query": query_text,   # For suggestions, text can be same as query
                "similarity_score": score,})

        response_data = {"sugested_queries": response_queries}
        return Response(response_data, status=status.HTTP_200_OK)



class ValidateWordAPIView(APIView):
    def get(self, request):
        word = request.GET.get("word", "").strip().lower()

        if not word:
            return Response({
                "word": "",
                "valid": False,
                "suggestions": []
            }, status=status.HTTP_200_OK)

        # Check if word is valid using your prefix-based suggestion
        suggestions = words_suggest(word, top_k=3)  # top 3 suggestions
        is_valid = word in suggestions

        response_data = {
            "word": word,
            "valid": is_valid,
            "suggestions": suggestions if not is_valid else []}

        return Response(response_data, status=status.HTTP_200_OK)



#  ---------------- For Postman and production -----------------------
class AnswerAPIView(APIView):
    def post(self, request):
        query = request.data.get("query", "").strip()
        if not query:
            return Response({"text": "", "images": []}, status=status.HTTP_200_OK)
        # 1️⃣ Get the RAG answer
        answer_text = get_answer_using_RAG(query)
        # 2️⃣ Check if images are needed
        show_images = needs_images(query)
        images = get_disease_images_from_query(query) if show_images else []
        # 3️⃣ Construct JSON response
        response_data = {
            "query": query,
            "text": answer_text,
            "images": images  # array of image URLs
        }
        return Response(response_data, status=status.HTTP_200_OK)


#  ---------------- For My Testing Purpose  -----------------------
# @method_decorator(csrf_exempt, name="dispatch")
# class AnswerAPIView(APIView):
#     def post(self, request):
#         query = request.data.get("query", "").strip()
#         # query = split_query_score(query)
#         if not query:
#             return StreamingHttpResponse("", content_type="text/plain")
#         def stream_response():
#             show_images = needs_images(query)
#             # 1️⃣ Header
#             header = f"✅ Answer for: {query}\n\n"
#             for ch in header:
#                 yield f"TEXT:{ch}\n"
#                 time.sleep(0.01)
#             yield f"\n"
#             # 2️⃣ RAG answer
#             answer = get_answer_using_RAG(query)
#             yield f"\n"
#             for ch in answer:
#                 yield f"TEXT:{ch}\n"
#                 time.sleep(0.003)
#             # 3️⃣ End text marker
#             yield "END_TEXT\n"
#             # 4️⃣ Images
#             if show_images:
#                 images = get_disease_images_from_query(query)
#                 if images:
#                     yield f"IMAGES:{json.dumps(images)}\n"
#         return StreamingHttpResponse(
#             stream_response(),
#             content_type="text/plain")




