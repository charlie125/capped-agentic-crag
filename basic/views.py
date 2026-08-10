import json
from django.shortcuts import render
from django.http import StreamingHttpResponse
from .forms import *
from .models import *
from rag_agent.naive_rag import naive_stream
from rag_agent.capped_crag import capped_stream
from rag_agent.uncapped_crag import uncapped_stream


def index(request):
    histories = UserQuery.objects.all()
    mode = request.GET.get("mode", "naive")
    form = QueryForm()
    return render(request, 'index.html', {'form': form, 'histories': histories, "mode": mode})


def stream_view(request):
    user_query = request.GET.get("q", "")
    mode = request.GET.get("mode", "naive")

    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    if mode == "capped":
        stream = capped_stream(user_query, session_id=session_id)
    elif mode == "uncapped":
        stream = uncapped_stream(user_query, session_id=session_id)
    else:
        stream = naive_stream(user_query)

    def sse_gen():
        full_response = ""
        try:
            for event in stream:
                if event["type"] == "thinking":
                    payload = json.dumps(
                        {"node": event["node"], "label": event["label"]})
                    yield f"event: thinking\ndata: {payload}\n\n"
                elif event["type"] == "token":
                    full_response += event["text"]
                    payload = json.dumps({"text": event["text"]})
                    yield f"event: token\ndata: {payload}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            UserQuery.objects.create(
                user_query=user_query, ai_response=full_response)
            yield "event: done\ndata: {}\n\n"

    resp = StreamingHttpResponse(sse_gen(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp
