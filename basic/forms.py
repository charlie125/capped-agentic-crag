from django import forms


class QueryForm(forms.Form):
    user_query = forms.CharField(max_length=150)
