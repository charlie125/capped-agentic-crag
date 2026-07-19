from django.urls import path
from . import views

app_name = "basic"

urlpatterns = [
    path("", views.index, name="index"),
    path("dashboard", views.dashboard, name="dashboard"),
]
