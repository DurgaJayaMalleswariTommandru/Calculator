from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
import subprocess
from django.conf import settings


def loginpage(request):
    if request.method == 'POST':
        username = request.POST.get('num1')
        password = request.POST.get('num2')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
    return render(request, 'login.html')

def registerpage(request):
    if request.method == 'POST':
        username = request.POST.get('num1')
        password = request.POST.get('num2')
        conform = request.POST.get('num3')
        if password != conform:
            return render(request, 'register.html', {'result': 'ERROR'})
        user = User.objects.create_user(username=username, password=password)
        return redirect('login')
    return render(request, 'register.html')

def home(request):
    return render(request, 'home.html')

def logoutpage(request):
    logout(request)
    return redirect('login')

def ML_results(request):
    try:
        # Run train.py script from ml folder
        result = subprocess.run(
            ['python', 'ML/train.py'],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse stdout for accuracy and precision
        accuracy = precision = None
        for line in result.stdout.splitlines():
            if line.startswith("ACCURACY="):
                accuracy = float(line.split("=")[1])
            elif line.startswith("PRECISION="):
                precision = float(line.split("=")[1])

        if accuracy is not None and precision is not None:
            return render(request, 'machine.html', {
                'accuracy': accuracy,
                'precision': precision
            })
        else:
            return render(request, 'machine.html', {
                'error': 'Could not parse output from train.py'
            })

    except subprocess.CalledProcessError:
        return render(request, 'machine.html', {
            'error': 'Failed to run train.py'
        })