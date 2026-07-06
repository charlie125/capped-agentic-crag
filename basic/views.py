from django.shortcuts import render
from .forms import QueryForm
from .models import *

# Create your views here.


def test(request):
    user_query = ''
    answering = []
    ai_answering = ''

    if request.method == "POST":
        form = QueryForm(request.POST)
        if form.is_valid():
            user_query = form.cleaned_data['query']
            answering = [val for key,
                         val in vector_db_search(user_query).items()]
            ai_answering = ai_respones(user_query)
    else:
        form = QueryForm()

    return render(request, 'test.html', {'form': form, 'user_query': user_query, 'answering': answering, 'ai_answering': ai_answering})
