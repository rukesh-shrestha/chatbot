"""mainprojectchat URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from pages import views

urlpatterns = [

    path('', views.home, name='home'),
    #path('chatbot/', include('chatbotAI.urls')),
    path('contact/', views.contact, name='contact'),
    path('about/', views.about, name='about'),
    path('register/', views.register, name='register'),
    path('price/', views.priceplan, name='priceplan'),
    path('logout/', views.logoutuser, name='logout'),
    #path('signin/register/', register),
    path('login/', views.loginuser, name='login'),
    path('admin/', admin.site.urls),
    path('get', views.chatbot, name='chatbot'),
    # path('accounts/',include('accounts.urls')),

]
