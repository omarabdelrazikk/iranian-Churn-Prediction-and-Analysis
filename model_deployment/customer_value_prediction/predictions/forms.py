from django import forms

class PredictionForm(forms.Form):
    call_failure = forms.FloatField()
    complains = forms.FloatField()
    subscription_length = forms.FloatField()
    charge_amount = forms.FloatField()
    frequency_of_sms = forms.FloatField()
    distinct_called_numbers = forms.FloatField()
    age_group = forms.FloatField()
    tariff_plan = forms.FloatField()
    status = forms.FloatField()
    average_of_use = forms.FloatField()
