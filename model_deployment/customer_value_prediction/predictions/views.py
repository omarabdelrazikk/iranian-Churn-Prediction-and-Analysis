import joblib
import numpy as np
from django.shortcuts import render
from .forms import PredictionForm
from sklearn.preprocessing import StandardScaler


# Load the models
regression_model = joblib.load('predictions/models/regression_model.pkl')
classification_model = joblib.load('predictions/models/classification_model.pkl')

def predict(request):
    regression_result = None
    classification_result = None
    form = PredictionForm(request.POST or None)
    
    if form.is_valid():
        features = np.array([[
            form.cleaned_data['call_failure'],
            form.cleaned_data['complains'],
            form.cleaned_data['subscription_length'],
            form.cleaned_data['charge_amount'],
            form.cleaned_data['frequency_of_sms'],
            form.cleaned_data['distinct_called_numbers'],
            form.cleaned_data['age_group'],
            form.cleaned_data['tariff_plan'],
            form.cleaned_data['status'],
            form.cleaned_data['average_of_use']
        ]])

        # Standardize the feature 'average_of_use'
        
        features[:, -1] = np.expm1((features[:, -1] * 0.24996399) + 0.68661658)
        #scaler.fit_transform(features[:, -1].reshape(-1, 1))

        # Predict using the regression and classification models
        regression_result = regression_model.predict(features)[0]
        classification_result = classification_model.predict(features)[0]

    return render(request, 'predict.html', {
        'form': form,
        'regression_result': regression_result,
        'classification_result': classification_result
    })