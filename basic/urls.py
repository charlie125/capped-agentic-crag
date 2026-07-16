from django.urls import path
from . import views

app_name = "basic"

urlpatterns = [
    path("", views.linear, name="linear"),
    path("uncapped", views.uncapped, name="uncapped"),
    path("capped", views.capped, name="capped"),

]
