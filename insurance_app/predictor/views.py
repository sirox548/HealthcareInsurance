import joblib
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd

# Load the model
model = joblib.load('insurance_model.pkl')

def predict_charges(request):
    if request.method == 'POST':
        # Get input data from the form
        age = float(request.POST.get('age'))
        sex = request.POST.get('sex')
        bmi = float(request.POST.get('bmi'))
        children = int(request.POST.get('children'))
        smoker = request.POST.get('smoker')
        region = request.POST.get('region')

        # Create a DataFrame from the input
        input_data = pd.DataFrame({
               'age': [age],
               'sex': [sex],
               'bmi': [bmi],
               'children': [children],
               'smoker': [smoker],
               'region': [region]
        })

        # Make a prediction
        prediction = model.predict(input_data)
        return JsonResponse({'prediction': prediction[0]})

    return render(request, 'predictor/predict.html')


