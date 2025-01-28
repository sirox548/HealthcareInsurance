from django.urls import path
from . import views

urlpatterns = [
       path('predict/', views.predict_charges, name='predict_charges'),
   ]