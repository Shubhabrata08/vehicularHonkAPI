from django.urls import path
from . import views
urlpatterns = [
    path('predictHonk',views.predictHonk,name='predictHonk'),
    # path('gpt2Test',views.gpt2Test,name='gpt2Test'),
]