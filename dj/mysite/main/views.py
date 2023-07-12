from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def index(response):
    return HttpResponse("looks like someone killed makise kurisu")

def v1(response):
    return HttpResponse('v1')