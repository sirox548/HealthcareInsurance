import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
import joblib

model = joblib.load('insurance_model.pkl')

def predict_charges(request):
    if request.method == 'POST':
        # Get input data
        age = float(request.POST.get('age'))
        sex = request.POST.get('sex')           # 'male' or 'female'
        bmi = float(request.POST.get('bmi'))
        children = int(request.POST.get('children'))
        smoker = request.POST.get('smoker')     # 'yes' or 'no'
        region = request.POST.get('region')     # e.g. 'northwest'

        # Encode categorical variables (must match training encoding)
        sex_encoded = 1 if sex.lower() == 'male' else 0
        smoker_encoded = 1 if smoker.lower() == 'yes' else 0

        # Create base input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex_encoded],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker_encoded],
            'region': [region]
        })

        # One-hot encode 'region' as expected by the model
        region_dummies = pd.get_dummies(input_data['region'], prefix='region')
        expected_region_columns = ['region_northwest', 'region_southeast', 'region_southwest']
        for col in expected_region_columns:
            if col not in region_dummies:
                region_dummies[col] = 0  # fill missing with 0

        input_data = input_data.drop('region', axis=1)
        input_data = pd.concat([input_data, region_dummies], axis=1)

        # Reorder columns to match model training order
        input_data = input_data.reindex(columns=[
            'age', 'sex', 'bmi', 'children', 'smoker',
            'region_northwest', 'region_southeast', 'region_southwest'
        ])

        # Make prediction
        # prediction = model.predict(input_data)
        prediction = round(float(model.predict(input_data)[0]), 2)

        # return JsonResponse({'prediction': float(prediction[0])})

    return render(request, 'predictor/predict.html', {'prediction': prediction})
