from django.urls import path
from .views import Word_AutocompleteAPIView, Query_Suggestions_APIView, AnswerAPIView,ImageDiseaseDetailsAPIView

urlpatterns = [
    path("word_autocomplete/", Word_AutocompleteAPIView.as_view(), name="autocomplete"),
    path("query_suggestion/", Query_Suggestions_APIView.as_view(), name="query_suggestion"),
    path("answer/", AnswerAPIView.as_view(), name="answer"),
    path("image-disease-details/",ImageDiseaseDetailsAPIView.as_view(),name="image_disease_details"),

]

