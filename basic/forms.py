from django import forms


class QueryForm(forms.Form):
    user_query = forms.CharField(required=True)

