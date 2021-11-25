# Importing the libraries
import requests
import os
from django.core import serializers
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse,HttpRequest
from django.shortcuts import render, redirect
from django.contrib.auth.models import User



from .models import Online,UploadForm,QuickForm,Offline
from .forms import CreateUserForm,UploadFileForm


from plotly.offline import plot
import plotly.graph_objs as go
import math
from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame
# Form Imports




def index(request):
    return render(request, 'index.html')


def login_view(request):
    if request.user.is_authenticated:
        return redirect('/')
    else:
        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)

            if user is not None:
                login(request, user)
                return redirect('/')
            else:
                messages.info(request, 'Username or password is incorrect')
        context = {}
        return render(request, 'login.html', context)


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('analysis')
    else:
        form = CreateUserForm()
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                messages.success(request, 'Account was created for ' + user)
                return redirect('login')

        context = {'form': form}
        return render(request, 'register.html', context)






@login_required(login_url='login')
def bridge(request):
    if request.method == 'POST':
        form = Online.objects.create(file = request.FILES['file'])
        form.email = request.user.email
        form.save()
        form = ""
        success = "Váš súbor bol úspešne odoslaný."
    else:
        form = UploadForm()
        success = ""
    context = {'serviceForm': form, 'success': success}
    return render(request, 'requirements.html', context)

# MODELS Apriori, UPC handled seperatly
@login_required(login_url='login')
def site_apriori(request):
    if request.method == 'POST':
        form = QuickForm(request.POST, request.FILES)
        if form.is_valid():
            # file is saved
            form.save()
            filename = request.FILES['file']
            # Data Preprocessing
            dataset = pd.read_csv('media/media/' + str(filename), header=None)
            rows = len(dataset.axes[0])
            cols = len(dataset.axes[1])
            transactions = []
            for i in range(0, int(rows)):
                transactions.append([str(dataset.values[i, j]) for j in range(0, int(cols))])

            # Training Apriori on the dataset
            from utils.apyori import apriori

            rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)


            # Well organised rules
            def inspect(results):
                lhs = [tuple(result[2][0][0])[0] for result in results]
                rhs = [tuple(result[2][0][1])[0] for result in results]
                #supports = [result[1] for result in results]
                #confidences = [result[2][0][2] for result in results]
                lifts = [result[2][0][3] for result in results]
                return list(zip(lhs, rhs, lifts))


            results: List[tuple] = list(rules)
            resultsinDataFrame = pd.DataFrame(inspect(results),
                                            columns=[' Left Hand Side ', ' Right Hand Side ', ' Lift '])

            vysledok: DataFrame = resultsinDataFrame.nlargest(n=int(rows), columns=' Lift ')
            vysledok[' Left Hand Side '].replace('nan', np.nan, inplace=True)
            vysledok.drop_duplicates(subset=[' Left Hand Side ', ' Right Hand Side '], inplace=True)
            vysledok.dropna(inplace=True)
            df = vysledok
            html = (
                df.style
                    .set_properties(**{'border-spacing': '1px'})
                    .render()
            )
    else:
        form = QuickForm()
        rows = ""
        cols = ""
        html = ""
        filename = ""
    context = {'table': html, 'form': form, 'rows':rows, 'cols':cols, 'filename': filename}
    return render(request, 'apriori.html', context)

@login_required(login_url='login')
def apriori_results(request):
    filename = request.FILES['file']
    # Data Preprocessing
    dataset = pd.read_csv('media/media/' + str(filename), header=None)
    rows = len(dataset.axes[0])
    cols = len(dataset.axes[1])
    transactions = []
    for i in range(0, int(rows)):
        transactions.append([str(dataset.values[i, j]) for j in range(0, int(cols))])

    # Training Apriori on the dataset
    from utils.apyori import apriori

    rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)


    # Well organised rules
    def inspect(results):
        lhs = [tuple(result[2][0][0])[0] for result in results]
        rhs = [tuple(result[2][0][1])[0] for result in results]
        #supports = [result[1] for result in results]
        #confidences = [result[2][0][2] for result in results]
        lifts = [result[2][0][3] for result in results]
        return list(zip(lhs, rhs, lifts))


    results: List[tuple] = list(rules)
    resultsinDataFrame = pd.DataFrame(inspect(results),
                                    columns=[' Left Hand Side ', ' Right Hand Side ', ' Lift '])

    vysledok: DataFrame = resultsinDataFrame.nlargest(n=int(rows), columns=' Lift ')
    vysledok[' Left Hand Side '].replace('nan', np.nan, inplace=True)
    vysledok.drop_duplicates(subset=[' Left Hand Side ', ' Right Hand Side '], inplace=True)
    vysledok.dropna(inplace=True)
    html = (
        vysledok.style
            .set_properties(**{'border-spacing': '1px'})
            .render()
    )

    context = {'table': html, 'filename':filename}
    return render(request, 'apresults.html', context)




@login_required(login_url='login')
def site_ucb(request):
    if request.method == 'POST':
        form = QuickForm(request.POST, request.FILES)
        if form.is_valid():
            # file is saved
            form.save()
            filename = request.FILES['file']
            dataset = pd.read_csv('media/media/' + str(filename))
            rows = len(dataset.axes[0]) 
            cols = len(dataset.axes[1])
            ads_selected = []
            numbers_of_selections = [0] * cols
            sums_of_rewards = [0] * cols
            total_reward = 0
            for n in range(1, rows): # Exclude Header
                ad = 0
                max_upper_bound = 0
                for i in range(0, cols):
                    if (numbers_of_selections[i] > 0):
                        average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                        delta_i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_selections[i])
                        upper_bound = average_reward + delta_i
                    else:
                        upper_bound = 1e400
                    if upper_bound > max_upper_bound:
                        max_upper_bound = upper_bound
                        ad = i
                ads_selected.append(ad)
                numbers_of_selections[ad] = numbers_of_selections[ad] + 1
                reward = dataset.values[n, ad]
                sums_of_rewards[ad] = sums_of_rewards[ad] + reward
                total_reward = total_reward + reward
            plot_div = plot([go.Histogram(x=ads_selected,
                                name='test',
                                opacity=0.8, marker_color='green')],
                    output_type='div')

    else:
        form = QuickForm()
        plot_div = ""
    context = {'form': form,'plot_div': plot_div}
    return render(request, 'upc.html', context)

@login_required(login_url='login')
def ucb_results(request):
    filename = request.FILES['file']
    dataset = pd.read_csv('media/media/' + str(filename))
    rows = len(dataset.axes[0]) 
    cols = len(dataset.axes[1])
    ads_selected = []
    numbers_of_selections = [0] * cols
    sums_of_rewards = [0] * cols
    total_reward = 0
    for n in range(1, rows): # Exclude header
        ad = 0
        max_upper_bound = 0
        for i in range(0, cols):
            if (numbers_of_selections[i] > 0):
                average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                delta_i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_selections[i])
                upper_bound = average_reward + delta_i
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
        ads_selected.append(ad)
        numbers_of_selections[ad] = numbers_of_selections[ad] + 1
        reward = dataset.values[n, ad]
        sums_of_rewards[ad] = sums_of_rewards[ad] + reward
        total_reward = total_reward + reward
    plot_div = plot([go.Histogram(x=ads_selected,
                        name='test',
                        opacity=0.8, marker_color='green')],
               output_type='div')
    context = {'plot_div': plot_div}
    return render(request, 'upcresults.html', context)




def previous(request):
    request.META['HTTP_REFERER']

def logout_view(request):
    logout(request)
    return redirect('login')
