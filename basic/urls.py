from django.urls import path
from . import views

app_name = "basic"

urlpatterns = [
    path("", views.index, name="index"),
    path("stream/", views.stream_view, name="stream_view"),
]
