from django.urls import path
from . import views  # Correct import for views in the same directory
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name="index"),
    path('runAnalysis', views.runAnalysis, name='run_analysis'),
    path('runVideoAnalysis', views.runVideoAnalysis, name='run_video_analysis'),
    path('getImages', views.getImages, name='get_images'),
    path('image-upload/', views.image_upload, name="image_upload"),
    path('video-upload/', views.video_upload, name="video_upload"),
    path('image/', views.image_analysis, name="image"),
    path('video/', views.video_analysis, name="video"),
    path('register/', views.register_view, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='authentication/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('history/', views.history_view, name='history'),
    path('history/delete/<int:analysis_id>/', views.delete_history_view, name='delete_history'),
    path('how-it-works/', views.how_it_works_view, name='how_it_works'),
    path('faqs/', views.faqs_view, name='faqs'),
    path('contact/', views.contact_view, name='contact'),
    path('send-message/', views.send_message_view, name='send_message'),
]

