from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib.auth.models import User,auth
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout

from .forms import UserForm

# Create your views here.
def home (request):

  #  return HttpResponse("<h1>Hello world my name is rukesh</h1>")
    return render(request,"base.html",{})
def contact (request,*args, **kwargs):
    return render(request,"contactus.html",{})

def about (request, *args, **kwargs):
   return render(request,"{#about}",{})

def priceplan (request, *args, **kwargs):
   return render(request,"priceplan.html",{})

def register (request):
    form = UserForm()
    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request,'Account created successfully..')
            return redirect('login')
    context = {'form': form}
    return render(request,"reg.html",context)

def loginuser(request):
    if request.method == 'POST':

        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request,username = username, password = password)
        if user is not None:
            login(request,user)
            return redirect('home')
        else:
            messages.info(request,"username $ password is incorrect...")
    context = {}
    return render(request,"log.html",context)

def logoutuser(request):
    logout(request)
    return redirect('home')
