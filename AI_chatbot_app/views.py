from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import StreamingHttpResponse
import re
import csv
import os
from datetime import datetime
import json
import time
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

# from .query_recommendation import recommend_titles
from .word_autocomplete import words_suggest
from .query_answer import get_answer_using_RAG, get_recommend_titles, split_query_score
from .query_image import needs_images, check_is_there_image, get_disease_images_from_query, get_disease_images
from utils.disease_service import get_disease_info
from utils.query_preprocess import detect_language
from utils.image_parser import extract_crop_disease_from_image
from utils.bangla_morphological_normalize import get_crop_name_for_image_path

class Word_AutocompleteAPIView(APIView):
    def get(self, request):
        q = request.GET.get("q", "").strip()
        # Invalid input
        if not q:
            return Response(
                {
                    "data": None,
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "Failed"
                },
                status=status.HTTP_400_BAD_REQUEST)
        # Get suggestions
        suggestions = words_suggest(q, top_k=5)
        # No suggestions found
        if not suggestions:
            return Response(
                {
                    "data": None,
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "Failed"
                },
                status=status.HTTP_400_BAD_REQUEST)
        # Success response
        return Response(
            {
                "data": suggestions,  # ["roof", "rooftop"]
                "status": status.HTTP_200_OK,
                "message": "Success"
            },
            status=status.HTTP_200_OK)


class Query_Suggestions_APIView(APIView):
    def post(self, request):
        selected_word = request.data.get("selected")
        # Validation
        if not selected_word:
            return Response(
                {
                    "data": None,
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "Failed"
                },
                status=status.HTTP_400_BAD_REQUEST)
        recommendations, similarity_scores = get_recommend_titles(selected_word)
        # If nothing found
        if not recommendations:
            return Response(
                {
                    "data": None,
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "Failed"
                },
                status=status.HTTP_400_BAD_REQUEST )
        # Extract only clean query texts
        data = []
        for rec_text in recommendations:
            query_text = rec_text.split("→")[0].strip()
            data.append(query_text)
        return Response(
            {
                "data": data,
                "status": status.HTTP_200_OK,
                "message": "Success"
            },
            status=status.HTTP_200_OK)



def save_query_answer_to_csv(query: str, answer: str):
    CSV_FILE_PATH = "/home/gflml/Chatbot/multi_modal_chatbot_new/query_answer_logs.csv"
    """
    Save query and answer to CSV with timestamp.
    Creates file with header if not exists.
    """
    file_exists = os.path.isfile(CSV_FILE_PATH)
    with open(CSV_FILE_PATH, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header only once
        if not file_exists:
            writer.writerow(["timestamp", "query", "answer"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            query,
            answer])


"""
How to test streaming properly:
Use curl in terminal:
curl -N -X POST http://192.168.50.175:8001/api/streaming_answer/ \
     -H "Content-Type: application/json" \
     -d '{"query": "What is a rooftop garden?"}'
"""
"""
curl -N -X POST http://127.0.0.1:8000/api/answer/ -H "Content-Type: application/json" -d '{"query": "half drum?"}'
"""

@method_decorator(csrf_exempt, name="dispatch")
class AnswerAPIView(APIView):
    def post(self, request):
        query = request.data.get("query", "").strip()
        # 1️⃣ Validate query
        if not query:
            return Response(
                {
                    "data": None,
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "Failed message"
                },
                status=status.HTTP_400_BAD_REQUEST)
        # 2️⃣ Get RAG answer
        answer, chunks, crop_name, disease_or_pest_name = get_answer_using_RAG(query)
        crop_name_for_image_path = []
        if len(crop_name) > 0:
            is_en_or_bn = detect_language(crop_name)
            if is_en_or_bn == 'bangla':
                crop_name_for_image_path = get_crop_name_for_image_path(crop_name)
                if len(chunks)>0 and chunks[0]['has_images']:
                    disease_or_pest_name = []
                    disease_or_pest_name.append(chunks[0]['disease_or_pest_name'])
            else:
                crop_name_for_image_path = crop_name
        # Save query & answer
        if answer:
            save_query_answer_to_csv(query, answer)
        # 3️⃣ If no answer found
        if not answer:
            return Response(
                {
                    "data": None,
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "Failed message"
                },
                status=status.HTTP_400_BAD_REQUEST)

        # 4️⃣ Check images
        images = []
        if chunks != "Low similarity" and chunks is not None:
            if check_is_there_image(chunks): # # check if there has image with that answer
                # show_images = needs_images(query)
                # show_images = True
                # if len(disease_or_pest_name) > 0:
                #     disease_or_pest_name = disease_or_pest_name[0]
                images = get_disease_images_from_query(chunks, crop_name_for_image_path, disease_or_pest_name)        
        # 5️⃣ Success response
        return Response(
            {
                "data": [
                    {
                        "text": answer,
                        "image": images  # always return list
                    }
                ],
                "status": status.HTTP_200_OK,
                "message": "Success"
            },
            status=status.HTTP_200_OK)


@method_decorator(csrf_exempt, name="dispatch")
class ImageDiseaseDetailsAPIView(APIView):
    def post(self, request):
        image_path = request.data.get("image_path")
        image_file = request.FILES.get("image")
        # ------------------------------
        # 1️⃣ Determine filename
        # ------------------------------
        if image_path:
            filename = os.path.basename(image_path)
        elif image_file:
            filename = image_file.name
        else:
            return Response(
                {
                    "data": None,
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "Provide image_path or image file"
                },
                status=status.HTTP_400_BAD_REQUEST )
        # ------------------------------
        # 2️⃣ Extract crop & disease
        # ------------------------------
        crop, disease = extract_crop_disease_from_image(filename)
        if not crop or not disease:
            return Response(
                {
                    "data": None,
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "Unable to extract crop or disease"
                },
                status=status.HTTP_400_BAD_REQUEST)
        # ------------------------------
        # 3️⃣ Fetch disease info
        # ------------------------------
        disease_info = get_disease_info(crop, disease)
        if not disease_info:
            return Response(
                {
                    "data": {
                        "crop": crop,
                        "disease": disease
                    },
                    "status": status.HTTP_404_NOT_FOUND,
                    "message": "No disease information found"
                },
                status=status.HTTP_404_NOT_FOUND )
        # ------------------------------
        # 4️⃣ Get ALL images for this disease
        # ------------------------------
        images = get_disease_images(crop, disease)
        # ------------------------------
        # 5️⃣ Success Response
        # ------------------------------
        return Response(
            {
                "data": {
                    **disease_info,
                    "images": images
                },
                "status": status.HTTP_200_OK,
                "message": "Success"
            },
            status=status.HTTP_200_OK)

