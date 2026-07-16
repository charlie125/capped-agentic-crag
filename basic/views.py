from django.shortcuts import render
from .forms import QueryForm
from .models import *
from rag_agent.linear_rag import linear_rag_respones

# Create your views here.


def linear(request):
    user_query = ""
    llm_answering = ""
    title = "linear"

    if request.method == "POST":
        form = QueryForm(request.POST)
        if form.is_valid():
            user_query = form.cleaned_data["query"]
            llm_answering = linear_rag_respones(user_query)

    else:
        form = QueryForm()

    return render(request, "test.html", {"form": form, "user_query": user_query, "llm_answering": llm_answering, "title": title})


def uncapped(request):
    user_query = ""
    llm_answering = ""
    title = "uncapped"

    if request.method == "POST":
        form = QueryForm(request.POST)
        if form.is_valid():
            user_query = form.cleaned_data["query"]
            llm_answering = linear_rag_respones(user_query)

    else:
        form = QueryForm()

    return render(request, "uncapped.html", {"form": form, "user_query": user_query, "llm_answering": llm_answering, "title": title})


def capped(request):
    user_query = ""
    llm_answering = ""
    title = "capped"

    if request.method == "POST":
        form = QueryForm(request.POST)
        if form.is_valid():
            user_query = form.cleaned_data["query"]
            llm_answering = linear_rag_respones(user_query)

    else:
        form = QueryForm()

    return render(request, "capped.html", {"form": form, "user_query": user_query, "llm_answering": llm_answering, "title": title})
