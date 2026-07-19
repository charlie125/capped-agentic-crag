from django.shortcuts import render
from .forms import *
from .models import *
from rag_agent.linear_rag import linear_rag_respones
from rag_agent.capped_agentic_rag import graph as capped_rag
from rag_agent.uncapped_agentic_rag import graph as uncapped_rag
import random

# Create your views here.


def index(request):
    sentence = random.choice(['How may I assist you today?', 'What can I do for you today?',
                              'Please let me know how I can be of assistance.', 'How can I help you today?', 'How can I help you out today?', 'How can I make your day easier today?'])
    ai_response = ""
    user_query = ""
    histories = UserQuery.objects.all()
    mode = request.GET.get("mode")

    if request.method == "POST":
        form = QueryForm(request.POST)
        if form.is_valid():
            user_query = form.cleaned_data['user_query']

            if mode == "linear":
                ai_response = linear_rag_respones(user_query=user_query)
            elif mode == "uncapped":
                ai_response = uncapped_rag.invoke({"messages": user_query})
            elif mode == "capped":
                ai_response = capped_rag.invoke(
                    {"messages": user_query, "rewrite_count": 0})

        UserQuery.objects.create(
            user_query=user_query, ai_mode=mode, ai_response=ai_answering)
    else:
        form = QueryForm()
    return render(request, 'index.html', {'form': form, 'histories': histories, 'sentence': sentence, "mode": mode})


def dashboard(request):
    pass
