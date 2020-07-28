from django import forms


class ClassificationForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea)

