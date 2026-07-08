from django.shortcuts import render
from .forms import QueryForm
from .models import *
from rag_agent.vector_db import vector_db_search

# Create your views here.


def test(request):
    user_query = ''
    vector_db_answering = ""
    llm_answering = 'LLM_test'

    if request.method == "POST":
        form = QueryForm(request.POST)
        if form.is_valid():
            user_query = form.cleaned_data['query']
            vector_db_answering = [
                val[0] for key, val in vector_db_search(user_query).items()]

    else:
        form = QueryForm()

    return render(request, 'test.html', {'form': form, 'user_query': user_query, 'vector_db_answering': vector_db_answering, 'llm_answering': llm_answering})
