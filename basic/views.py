from django.shortcuts import render
from .forms import QueryForm
from .models import *

# Create your views here.


def test(request):
    user_query = ''
    db_answering = []
    llm_answering = ''

    if request.method == "POST":
        form = QueryForm(request.POST)
        if form.is_valid():
            user_query = form.cleaned_data['query']
    else:
        form = QueryForm()

    return render(request, 'test.html', {'form': form, 'user_query': user_query, 'db_answering': db_answering, 'llm_answering': llm_answering})
