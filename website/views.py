import datetime
from django.shortcuts import render, redirect, HttpResponseRedirect
import asyncio
from multiprocessing import Pool
import numpy as np
import subprocess
from django.shortcuts import render
from django.conf import settings
import streamlit as st
import sys
import os
from website.ImageForgeryDetection.FakeImageDetector import FID
##from website.videoForgeryDetection.videoFunctions import *
from django.core.files.storage import FileSystemStorage

import website.ImageForgeryDetection.double_jpeg_compression as djc  # ADD1
import website.ImageForgeryDetection.noise_variance as nvar
import website.ImageForgeryDetection.copy_move_cfa as cfa
import website.ImageForgeryDetection.copy_move_sift as sift

from optparse import OptionParser
from json import dumps
from urllib.parse import unquote
from website.VideoForgeryDetection.detect_video import detect_video_forgery
from PIL import Image
from PIL.ExifTags import TAGS
from django.contrib.auth.decorators import login_required # Add this import
from django.views.decorators.http import require_POST # Add this import if not already present
from django.contrib import messages # Add this import if not already present
from django.shortcuts import get_object_or_404 # Add this import if not already present
from django.contrib.auth import login # Add this import for the register_view
from .forms import RegistrationForm # Add this line to import your form

# Import your models if they are defined elsewhere
# from .models import AnalysisRecord # Assuming you have models.py

# Create your views here.

fileurl = ''
inputImageUrl = ''
result = {}
inputVideoUrl = ''
fileVideoUrl = ''
infoDict = {}
inputImage=''


def getMetaData(path):
    global infoDict
    # CODE for metadata starts
    try:
        imgPath = path
        exeProcess = "hachoir-metadata"
        process = subprocess.Popen([exeProcess, imgPath],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)

        for tag in process.stdout:
            line = tag.strip().split(':')
            if len(line) >= 2:  # Ensure the line has at least two parts
                infoDict[line[0].strip()] = line[-1].strip()

        # If no metadata was extracted, add basic file information
        if not infoDict:
            # Add basic file information
            if os.path.exists(path):
                # Get file size
                file_size_bytes = os.path.getsize(path)
                if file_size_bytes < 1024 * 1024:  # Less than 1MB
                    file_size = f"{round(file_size_bytes / 1024, 2)} KB"
                else:
                    file_size = f"{round(file_size_bytes / (1024 * 1024), 2)} MB"
                infoDict["File Size"] = file_size
                
                # Get file creation and modification dates
                creation_time = os.path.getctime(path)
                modification_time = os.path.getmtime(path)
                infoDict["Creation Date"] = datetime.datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
                infoDict["Last Modified"] = datetime.datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d %H:%M:%S')
                
                # Get file extension
                _, file_extension = os.path.splitext(path)
                infoDict["File Type"] = file_extension.upper().replace('.', '')
                
                # Try to get image dimensions using PIL
                try:
                    with Image.open(path) as img:
                        width, height = img.size
                        infoDict["Dimensions"] = f"{width} × {height}"
                        infoDict["Format"] = img.format
                        infoDict["Mode"] = img.mode
                except Exception as e:
                    print(f"Error getting image dimensions: {e}")

        for k, v in infoDict.items():
            print(k, ':', v)
        if "Metadata" in infoDict.keys():
            del infoDict["Metadata"]
    except Exception as e:
        print(f"Error in getMetaData: {e}")
        # Ensure we have at least some basic metadata even if the extraction fails
        infoDict["File Path"] = path
        infoDict["Error"] = f"Could not extract full metadata: {str(e)}"
    # CODE for metadata ends


# Add these imports if not already present
import datetime
import subprocess
import os
from urllib.parse import unquote
from django.conf import settings

# Update the global variables section to include video_metadata
fileurl = ''
inputImageUrl = ''
result = {}
inputVideoUrl = ''
fileVideoUrl = ''
infoDict = {}
inputImage = ''
video_metadata = {}  # Add this global variable

# Your existing getMetaData function remains unchanged

# Enhanced video metadata extraction function
def get_video_metadata(filename):
    """
    Extract metadata from video files using multiple tools
    Returns a dictionary of video properties
    """
    properties = {}
    
    try:
        # Try to use ffprobe for detailed metadata (more reliable than hachoir for videos)
        try:
            import json
            import subprocess
            
            # Get detailed video information using ffprobe
            ffprobe_cmd = [
                'ffprobe', 
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                filename
            ]
            
            ffprobe_output = subprocess.check_output(ffprobe_cmd).decode('utf-8')
            ffprobe_data = json.loads(ffprobe_output)
            
            # Extract format information
            if 'format' in ffprobe_data:
                format_data = ffprobe_data['format']
                
                # Basic file information
                if 'filename' in format_data:
                    properties['file name'] = os.path.basename(format_data['filename'])
                
                if 'size' in format_data:
                    size_bytes = int(format_data['size'])
                    if size_bytes < 1024 * 1024:  # Less than 1MB
                        properties['file size'] = f"{size_bytes / 1024:.2f} KB"
                    elif size_bytes < 1024 * 1024 * 1024:  # Less than 1GB
                        properties['file size'] = f"{size_bytes / (1024 * 1024):.2f} MB"
                    else:
                        properties['file size'] = f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
                
                if 'format_name' in format_data:
                    properties['container format'] = format_data['format_name']
                
                if 'format_long_name' in format_data:
                    properties['format'] = format_data['format_long_name']
                
                if 'bit_rate' in format_data:
                    bit_rate = int(format_data['bit_rate']) / 1000
                    properties['bit rate'] = f"{bit_rate:.2f} kbps"
                
                if 'duration' in format_data:
                    duration_sec = float(format_data['duration'])
                    minutes, seconds = divmod(duration_sec, 60)
                    hours, minutes = divmod(minutes, 60)
                    properties['duration'] = f"{int(hours):02d}:{int(minutes):02d}:{seconds:.2f}"
                
                # Extract tags if available
                if 'tags' in format_data:
                    tags = format_data['tags']
                    
                    if 'creation_time' in tags:
                        properties['creation date'] = tags['creation_time']
                    
                    if 'title' in tags:
                        properties['title'] = tags['title']
                    
                    if 'artist' in tags:
                        properties['artist'] = tags['artist']
                    
                    if 'album' in tags:
                        properties['album'] = tags['album']
                    
                    if 'comment' in tags:
                        properties['comment'] = tags['comment']
                    
                    if 'encoder' in tags:
                        properties['encoder'] = tags['encoder']
                    
                    # Add any other tags that might be useful
                    for tag, value in tags.items():
                        if tag not in ['creation_time', 'title', 'artist', 'album', 'comment', 'encoder']:
                            properties[tag] = value
            
            # Extract stream information
            if 'streams' in ffprobe_data:
                video_stream = None
                audio_stream = None
                
                # Find the first video and audio streams
                for stream in ffprobe_data['streams']:
                    if stream.get('codec_type') == 'video' and not video_stream:
                        video_stream = stream
                    elif stream.get('codec_type') == 'audio' and not audio_stream:
                        audio_stream = stream
                
                # Process video stream
                if video_stream:
                    if 'codec_name' in video_stream:
                        properties['video codec'] = video_stream['codec_name']
                    
                    if 'codec_long_name' in video_stream:
                        properties['video codec full name'] = video_stream['codec_long_name']
                    
                    if 'width' in video_stream and 'height' in video_stream:
                        properties['resolution'] = f"{video_stream['width']} × {video_stream['height']} pixels"
                        properties['width'] = f"{video_stream['width']} pixels"
                        properties['height'] = f"{video_stream['height']} pixels"
                    
                    if 'display_aspect_ratio' in video_stream:
                        properties['aspect ratio'] = video_stream['display_aspect_ratio']
                    
                    if 'r_frame_rate' in video_stream:
                        frame_rate = video_stream['r_frame_rate'].split('/')
                        if len(frame_rate) == 2 and int(frame_rate[1]) != 0:
                            fps = int(frame_rate[0]) / int(frame_rate[1])
                            properties['frame rate'] = f"{fps:.2f} fps"
                    
                    if 'bits_per_raw_sample' in video_stream:
                        properties['bits per pixel'] = f"{video_stream['bits_per_raw_sample']} bits"
                    
                    if 'pix_fmt' in video_stream:
                        properties['pixel format'] = video_stream['pix_fmt']
                    
                    if 'color_space' in video_stream:
                        properties['color space'] = video_stream['color_space']
                    
                    if 'color_range' in video_stream:
                        properties['color range'] = video_stream['color_range']
                    
                    if 'color_transfer' in video_stream:
                        properties['color transfer'] = video_stream['color_transfer']
                    
                    if 'color_primaries' in video_stream:
                        properties['color primaries'] = video_stream['color_primaries']
                    
                    # Calculate total frames if possible
                    if 'duration' in video_stream and 'r_frame_rate' in video_stream:
                        try:
                            duration = float(video_stream['duration'])
                            frame_rate = video_stream['r_frame_rate'].split('/')
                            if len(frame_rate) == 2 and int(frame_rate[1]) != 0:
                                fps = int(frame_rate[0]) / int(frame_rate[1])
                                total_frames = int(duration * fps)
                                properties['total frames'] = f"{total_frames}"
                        except:
                            pass
                
                # Process audio stream
                if audio_stream:
                    if 'codec_name' in audio_stream:
                        properties['audio codec'] = audio_stream['codec_name']
                    
                    if 'codec_long_name' in audio_stream:
                        properties['audio codec full name'] = audio_stream['codec_long_name']
                    
                    if 'sample_rate' in audio_stream:
                        sample_rate = int(audio_stream['sample_rate'])
                        properties['audio sample rate'] = f"{sample_rate / 1000:.1f} kHz"
                    
                    if 'channels' in audio_stream:
                        properties['audio channels'] = audio_stream['channels']
                    
                    if 'channel_layout' in audio_stream:
                        properties['audio channel layout'] = audio_stream['channel_layout']
                    
                    if 'bit_rate' in audio_stream:
                        audio_bit_rate = int(audio_stream['bit_rate']) / 1000
                        properties['audio bit rate'] = f"{audio_bit_rate:.2f} kbps"
        except Exception as e:
            print(f"Error getting ffprobe data: {e}")
            
            # Fall back to hachoir-metadata if ffprobe fails
            try:
                result = subprocess.Popen(['hachoir-metadata', filename, '--raw', '--level=9'],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                
                results = result.stdout.read().decode('utf-8').split('\r\n')
                
                # Process all metadata lines
                for item in results:
                    if ':' in item:
                        parts = item.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip().replace('- ', '')
                            value = parts[1].strip()
                            properties[key] = value
                
                # Extract specific properties with better formatting
                for item in results:
                    if item.startswith('- duration: '):
                        duration = item.lstrip('- duration: ')
                        if '.' in duration:
                            t = datetime.datetime.strptime(duration, '%H:%M:%S.%f')
                        else:
                            t = datetime.datetime.strptime(duration, '%H:%M:%S')
                        seconds = (t.microsecond / 1e6) + t.second + (t.minute * 60) + (t.hour * 3600)
                        properties['duration'] = f"{round(seconds)} seconds"
                    
                    if item.startswith('- width: '):
                        properties['width'] = f"{int(item.lstrip('- width: '))} pixels"
                    
                    if item.startswith('- height: '):
                        properties['height'] = f"{int(item.lstrip('- height: '))} pixels"
            except Exception as e:
                print(f"Error getting hachoir-metadata: {e}")
        
        # Add file information if not already present
        if 'file name' not in properties and os.path.exists(filename):
            properties['file name'] = os.path.basename(filename)
        
        if 'file size' not in properties and os.path.exists(filename):
            file_size_bytes = os.path.getsize(filename)
            if file_size_bytes < 1024 * 1024:  # Less than 1MB
                file_size = f"{round(file_size_bytes / 1024, 2)} KB"
            elif file_size_bytes < 1024 * 1024 * 1024:  # Less than 1GB
                file_size = f"{round(file_size_bytes / (1024 * 1024), 2)} MB"
            else:
                file_size = f"{round(file_size_bytes / (1024 * 1024 * 1024), 2)} GB"
            properties['file size'] = file_size
        
        if 'creation date' not in properties and os.path.exists(filename):
            creation_time = os.path.getctime(filename)
            properties['creation date'] = datetime.datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
        
        if 'last modified' not in properties and os.path.exists(filename):
            modification_time = os.path.getmtime(filename)
            properties['last modified'] = datetime.datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d %H:%M:%S')
        
        if 'file type' not in properties and os.path.exists(filename):
            _, file_extension = os.path.splitext(filename)
            properties['file type'] = file_extension.upper().replace('.', '')
        
        # Clean up properties
        for key in list(properties.keys()):
            if not properties[key] or properties[key] == "None" or 'err!' in str(properties[key]):
                del properties[key]
        
        return properties
    except Exception as e:
        print(f"Error extracting video metadata: {e}")
        # Return basic file information if metadata extraction fails
        basic_info = {}
        if os.path.exists(filename):
            basic_info['file name'] = os.path.basename(filename)
            
            file_size_bytes = os.path.getsize(filename)
            if file_size_bytes < 1024 * 1024:  # Less than 1MB
                file_size = f"{round(file_size_bytes / 1024, 2)} KB"
            elif file_size_bytes < 1024 * 1024 * 1024:  # Less than 1GB
                file_size = f"{round(file_size_bytes / (1024 * 1024), 2)} MB"
            else:
                file_size = f"{round(file_size_bytes / (1024 * 1024 * 1024), 2)} GB"
            basic_info['file size'] = file_size
            
            creation_time = os.path.getctime(filename)
            modification_time = os.path.getmtime(filename)
            basic_info['creation date'] = datetime.datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
            basic_info['last modified'] = datetime.datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d %H:%M:%S')
            
            _, file_extension = os.path.splitext(filename)
            basic_info['file type'] = file_extension.upper().replace('.', '')
        
        return basic_info

# Update the runVideoAnalysis function to use the enhanced metadata
def runVideoAnalysis(request):
    global fileVideoUrl, inputVideoUrl, result, video_metadata
    
    # Initialize variables if they don't exist
    if 'fileVideoUrl' not in globals():
        fileVideoUrl = ''
    if 'inputVideoUrl' not in globals():
        inputVideoUrl = ''
    if 'result' not in globals():
        result = None
    if 'video_metadata' not in globals():
        video_metadata = {}
    
    # Check if we're submitting a new video or using an existing one
    if request.POST.get('run'):
        # Reset results when uploading a new video
        result = None
        video_metadata = {}
        
        inputVideo = request.FILES['input_video'] if 'input_video' in request.FILES else None
        if inputVideo:
            fs = FileSystemStorage()
            file = fs.save(inputVideo.name, inputVideo)
            fileVideoUrl = fs.path(file)
            fileVideoUrl = os.path.normpath(fileVideoUrl)
            fileVideoUrl = unquote(fileVideoUrl)
            inputVideoUrl = fs.url(file)  # Use URL for template
            print('fileVideoUrl---------------------------', fileVideoUrl)
            
            # Get video metadata
            try:
                video_metadata = get_video_metadata(fileVideoUrl)
                print("Video metadata:", video_metadata)
            except Exception as e:
                print(f"Error getting video metadata: {e}")
                video_metadata = {}
    
    # Process the video if we have a valid file and no results yet
    if request.POST.get('detect') and fileVideoUrl and not result:
        try:
            detection_result = detect_video_forgery(fileVideoUrl)
            print("Detection result:", detection_result)
            
            if detection_result.get('result') in ['Authentic', 'Forged']:
                result = {
                    'type': detection_result.get('result'),
                    'forged_frames': detection_result.get('f_frames', 0)
                }
            else:
                error_message = detection_result.get('message', 'Unknown error during video analysis')
                return render(request, "detection/video.html", {
                    'error': error_message,
                    'input_video': inputVideoUrl,
                    'video_metadata': video_metadata
                })
                
        except Exception as e:
            print(f"Error in video forgery detection: {e}")
            return render(request, "detection/video.html", {
                'error': f"Error analyzing video: {str(e)}",
                'input_video': inputVideoUrl,
                'video_metadata': video_metadata
            })
    
    # Render the template with all available data
    return render(request, "detection/video.html", {
        'result': result,
        'input_video': inputVideoUrl,
        'video_metadata': video_metadata
    })

# Also update the video_upload function to use the enhanced metadata
@login_required
def video_upload(request):
    global fileVideoUrl, inputVideoUrl, result, video_metadata
    
    if request.method == 'POST':
        inputVideo = request.FILES.get('video')
        if inputVideo:
            # Reset previous results
            result = None
            video_metadata = {}
            
            # Save the uploaded video
            fs = FileSystemStorage()
            file = fs.save(inputVideo.name, inputVideo)
            fileVideoUrl = fs.path(file)
            fileVideoUrl = os.path.normpath(fileVideoUrl)
            fileVideoUrl = unquote(fileVideoUrl)
            inputVideoUrl = fs.url(file)
            
            # Get video metadata
            try:
                video_metadata = get_video_metadata(fileVideoUrl)
                print("Video metadata:", video_metadata)
            except Exception as e:
                print(f"Error getting video metadata: {e}")
                video_metadata = {}
            
            # Directly analyze the video
            try:
                detection_result = detect_video_forgery(fileVideoUrl)
                print("Detection result:", detection_result)
                
                if detection_result.get('result') in ['Authentic', 'Forged']:
                    result = {
                        'type': detection_result.get('result'),
                        'forged_frames': detection_result.get('f_frames', 0)
                    }
                else:
                    error_message = detection_result.get('message', 'Unknown error during video analysis')
                    return render(request, "detection/video.html", {
                        'error': error_message,
                        'input_video': inputVideoUrl,
                        'video_metadata': video_metadata
                    })
                    
            except Exception as e:
                print(f"Error in video forgery detection: {e}")
                return render(request, "detection/video.html", {
                    'error': f"Error analyzing video: {str(e)}",
                    'input_video': inputVideoUrl,
                    'video_metadata': video_metadata
                })
            
            # Redirect to video analysis page with results
            return render(request, "detection/video.html", {
                'result': result,
                'input_video': inputVideoUrl,
                'video_metadata': video_metadata
            })
            
    return render(request, 'detection/video_upload.html')

# Update the video_analysis function to ensure metadata is properly passed
def video_analysis(request):
    global fileVideoUrl, inputVideoUrl, result, video_metadata
    
    # Initialize context with available data
    context = {
        'result': result if 'result' in globals() else None,
        'input_video': inputVideoUrl if 'inputVideoUrl' in globals() else None,
        'video_metadata': video_metadata if 'video_metadata' in globals() else {}
    }
    
    return render(request, "detection/video.html", context)


def index(request):
    return render(request, "index.html")


def video(request):
    return render(request, "video.html")


def image(request):
    return render(request, "image.html")

# Removed the pdf view function
# def pdf(request):
#     return render(request, "pdf.html")

# Removed the runPdf2image view function
# def runPdf2image(request):
#     global filePdfUrl, inputPdfUrl
#     if request.POST.get('run'):
#         inputPdf = request.FILES['input_pdf'] if 'input_pdf' in request.FILES else None
#         if inputPdf:
#             fs = FileSystemStorage()
#             file = fs.save(inputPdf.name, inputPdf)
#             fileurl = fs.url(file)
#             inputPdfUrl = '../media/' + inputPdf.name
#             fileurl = os.getcwd() + '/media/' + inputPdf.name
#             images = convert_from_path(fileurl)
#             imageurl = []
#             pdfImagesResults=[]
#             for i in range(len(images)):
#                 # Save pages as images in the pdf
#                 images[i].save(fileurl.strip(".pdf") + 'page' + str(i) + '.jpg', 'JPEG')
#                 #This list is used to generate table on pdf.html
#                 pageName=inputPdf.name.strip(".pdf") + 'page' + str(i) + '.jpg'
#                 imageurl.append('../media/' + pageName)
#                 imagefileurl = os.getcwd()  +'/media/'+pageName
#                 res = FID().predict_result(imagefileurl)
#                 result = {'type': res[0], 'confidence': res[1]}
#                 pdfImagesResults.append(result)
#             res=zip(imageurl,pdfImagesResults)

#         return render(request, "pdf.html", {'input_pdf': inputPdfUrl, 'pdf_img': res,})

#     if request.POST.get('passImage'):
#             global inputImageUrl, inputImage
#             inputImage=''
#             counter = request.POST.get('passImage')
#             inputImageUrl = request.POST.get('image_url-'+counter)
#             return render(request, "image.html",{'input_image': inputImageUrl,})



# Add this import at the top of the file
# Remove the import for AnalysisRecord
# from .models import AnalysisRecord

# In the runAnalysis function, remove the code that saves analysis records
def runAnalysis(request):
    global fileurl, inputImageUrl, result, infoDict, inputImage

    if request.method == 'POST':
        inputImg = request.FILES.get('image')
        if inputImg:
            fs = FileSystemStorage()
            file = fs.save(inputImg.name, inputImg)
            file_path = fs.path(file)
            fileurl = os.path.normpath(unquote(file_path))  # absolute path for processing
            inputImage = fs.url(file)  # URL for rendering in templates
            print('Saved file path:', fileurl)
            print('File URL for template:', inputImage)
        else:
            result = "Error"
            infoDict = {"message": "Please upload an image first."}
            return render(request, "detection/image.html", {
                'result': result,
                'infoDict': infoDict,
                'input_image': '',
                'ela_url': None
            })

        if not os.path.exists(fileurl):
            print(f"Error: File does not exist: {fileurl}")
            result = "Error"
            infoDict = {"message": f"File not found at path: {fileurl}"}
            return render(request, "detection/image.html", {
                'result': result,
                'infoDict': infoDict,
                'input_image': inputImage,
                'ela_url': None
            })

        # Extract metadata
        infoDict.clear()
        getMetaData(fileurl)
        
        # Ensure we have at least some basic metadata
        if not infoDict:
            # Add basic file information if metadata extraction failed
            infoDict["File Name"] = os.path.basename(fileurl)
            infoDict["File Path"] = fileurl
            infoDict["Analysis Date"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert metadata to a list of tuples for the template
        metadata_list = [(key, value) for key, value in infoDict.items()]
        
        # Run detection
        try:
            res = FID().predict_result(fileurl)
        except Exception as e:
            print(f"Error in FID prediction: {e}")
            result = {'type': 'Error', 'confidence': 0}
            return render(request, "detection/image.html", {
                'result': result,
                'input_image': inputImage,
                'metadata': metadata_list,  # Use the list of tuples instead of dict.items()
                'ela_url': None,
                'detection_method': "CNN-based Classification with ELA",
                'verification_text': "Error during analysis"
            })

        # Prepare ELA image
        ela_url = None
        try:
            temp_dir = 'temp'
            media_temp_dir = os.path.join(settings.MEDIA_ROOT, temp_dir)
            os.makedirs(media_temp_dir, exist_ok=True)
            output_path = os.path.join(media_temp_dir, 'output.jpg')
            FID().show_ela(fileurl, save_path=output_path)
            ela_url = f"{settings.MEDIA_URL}{temp_dir}/output.jpg"
        except Exception as e:
            print(f"Error generating ELA image: {e}")

        # Valid result
        if res[0] in ['Authentic', 'Forged']:
            result = {'type': res[0], 'confidence': res[1]}
        else:
            result = {'type': 'Unknown', 'confidence': 0}

        # Set the initial detection method
        detection_method = "CNN-based Classification with ELA"
        verification_text = "Multiple forensic techniques applied"
        
        # Print debug info
        print("Metadata items:", metadata_list)
        print("Result:", result)
        print("Detection method:", detection_method)
        
        return render(request, "detection/image.html", {
            'result': result,
            'input_image': inputImage,
            'metadata': metadata_list,  # Ensure metadata is a list
            'ela_url': ela_url,
            'detection_method': detection_method,
            'verification_text': verification_text
        })



def getImages(request):
    global fileurl, inputImageUrl, result, inputImage, infoDict
    
    temp_dir = 'temp'
    media_temp_dir = os.path.join(settings.MEDIA_ROOT, temp_dir)
    os.makedirs(media_temp_dir, exist_ok=True)
    
    output_path = os.path.join(media_temp_dir, 'output.jpg')
    output_url = f"{settings.MEDIA_URL}{temp_dir}/output.jpg"
    
    # Check if fileurl is valid
    if not fileurl or not os.path.exists(fileurl):
        error = f"Input image path does not exist: {fileurl}"
        print(error)
        return render(request, "detection/image.html", {
            'error': error, 
            'input_image': inputImage, 
            'result': result,
            'metadata': list(infoDict.items()) if infoDict else [("File Error", "No valid file path")]
        })
    
    # Ensure we have metadata
    if not infoDict:
        getMetaData(fileurl)
        
        # If still no metadata, add basic file information
        if not infoDict:
            infoDict["File Name"] = os.path.basename(fileurl)
            infoDict["File Path"] = fileurl
            infoDict["Analysis Date"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Check which button was pressed and generate the appropriate image
    detection_method = "CNN-based Classification with ELA"  # Default method
    verification_text = "Multiple forensic techniques applied"
    
    try:
        if request.POST.get('ela'):
            FID().show_ela(fileurl, save_path=output_path)
            detection_method = "Error Level Analysis (ELA)"
            verification_text = "Inconsistent compression artifacts detected" if result.get('type') == "Forged" else "Multiple forensic techniques applied"
        elif request.POST.get('mask'):
            FID().genMask(fileurl, save_path=output_path)
            detection_method = "Deep Learning Segmentation"
            verification_text = "Manipulated regions highlighted in mask" if result.get('type') == "Forged" else "Multiple forensic techniques applied"
        elif request.POST.get('edge_map'):
            FID().detect_edges(fileurl, save_path=output_path)
            detection_method = "Edge Detection Analysis"
            verification_text = "Edge inconsistencies detected" if result.get('type') == "Forged" else "Multiple forensic techniques applied"
        elif request.POST.get('na'):
            FID().apply_na(fileurl, save_path=output_path)
            detection_method = "Noise Variance Analysis"
            verification_text = "Noise pattern inconsistencies detected" if result.get('type') == "Forged" else "Multiple forensic techniques applied"
        elif request.POST.get('copy_move_sift'):
            # Import the SIFT module directly
            import website.ImageForgeryDetection.copy_move_sift as sift
            
            # Create a SIFT detector instance with the input and output paths
            sift_detector = sift.CopyMoveSIFT(fileurl, output_path)
            detection_method = "SIFT Feature Matching"
            verification_text = "Duplicated regions connected by lines" if result.get('type') == "Forged" else "Multiple forensic techniques applied"
        else:
            # Default to ELA if no specific button was pressed
            FID().show_ela(fileurl, save_path=output_path)
            detection_method = "Error Level Analysis (ELA)"
            verification_text = "Multiple forensic techniques applied"
            
        # Force a timestamp on the URL to prevent browser caching
        timestamp = int(datetime.datetime.now().timestamp())
        output_url = f"{output_url}?t={timestamp}"
        
        # Print debug info
        print("Metadata items in getImages:", list(infoDict.items()))
        print("Result in getImages:", result)
        print("Detection method in getImages:", detection_method)
        
        return render(request, "detection/image.html", {
            'ela_url': output_url, 
            'input_image': inputImage, 
            'result': result,
            'metadata': list(infoDict.items()),  # Ensure metadata is a list
            'detection_method': detection_method,
            'verification_text': verification_text
        })
        
    except Exception as e:
        print(f"Error generating analysis: {e}")
        return render(request, "detection/image.html", {
            'error': str(e), 
            'input_image': inputImage, 
            'result': result,
            'metadata': list(infoDict.items()) if infoDict else [("Error", str(e))],  # Ensure metadata is a list
            'detection_method': "CNN-based Classification with ELA",
            'verification_text': "Error during analysis"
        })

    # Remove the redundant code below as it's unreachable after the return statement above
    # The if statements for mask, edge_map, na, and copy_move_sift are already handled in the try block



# Add the runVideoAnalysis view function
def runVideoAnalysis(request):
    global fileVideoUrl, inputVideoUrl, result, video_metadata
    
    # Initialize variables if they don't exist
    if 'fileVideoUrl' not in globals():
        fileVideoUrl = ''
    if 'inputVideoUrl' not in globals():
        inputVideoUrl = ''
    if 'result' not in globals():
        result = None
    if 'video_metadata' not in globals():
        video_metadata = {}
    
    # Check if we're submitting a new video or using an existing one
    if request.POST.get('run'):
        # Reset results when uploading a new video
        result = None
        video_metadata = {}
        
        inputVideo = request.FILES['input_video'] if 'input_video' in request.FILES else None
        if inputVideo:
            fs = FileSystemStorage()
            file = fs.save(inputVideo.name, inputVideo)
            fileVideoUrl = fs.path(file)
            fileVideoUrl = os.path.normpath(fileVideoUrl)
            fileVideoUrl = unquote(fileVideoUrl)
            inputVideoUrl = fs.url(file)  # Use URL for template
            print('fileVideoUrl---------------------------', fileVideoUrl)
            
            # Get video metadata
            try:
                video_metadata = get_video_metadata(fileVideoUrl)
                print("Video metadata:", video_metadata)
            except Exception as e:
                print(f"Error getting video metadata: {e}")
                video_metadata = {}
    
    # Process the video if we have a valid file and no results yet
    if request.POST.get('detect') and fileVideoUrl and not result:
        try:
            detection_result = detect_video_forgery(fileVideoUrl)
            print("Detection result:", detection_result)
            
            if detection_result.get('result') in ['Authentic', 'Forged']:
                result = {
                    'type': detection_result.get('result'),
                    'forged_frames': detection_result.get('f_frames', 0)
                }
                
                # Remove the code that saves analysis records
                # if request.user.is_authenticated:
                #     AnalysisRecord.objects.create(
                #         user=request.user,
                #         type='video',
                #         result=detection_result.get('result'),
                #         file_path=fileVideoUrl
                #     )
                #     messages.success(request, "Analysis record saved to your history.")
            else:
                error_message = detection_result.get('message', 'Unknown error during video analysis')
                return render(request, "detection/video.html", {
                    'error': error_message,
                    'input_video': inputVideoUrl,
                    'video_metadata': video_metadata
                })
                
        except Exception as e:
            print(f"Error in video forgery detection: {e}")
            return render(request, "detection/video.html", {
                'error': f"Error analyzing video: {str(e)}",
                'input_video': inputVideoUrl,
                'video_metadata': video_metadata
            })
    
    # Render the template with all available data
    return render(request, "detection/video.html", {
        'result': result,
        'input_video': inputVideoUrl,
        'video_metadata': video_metadata
    })


# Registration View
def register_view(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST) # Now this should work
        if form.is_valid():
            user = form.save()
            login(request, user) # Log the user in directly after registration
            messages.success(request, 'Registration successful! Welcome.')
            return redirect('index') # Redirect to home page (assuming 'index' is your home page name)
        else:
            # Add form errors to messages
            for field, errors in form.errors.items():
                 for error in errors:
                      messages.error(request, f"{field}: {error}")
    else:
        form = RegistrationForm() # This should also work now
    return render(request, 'authentication/register.html', {'form': form})

# Profile View (requires login)
@login_required
def profile_view(request):
    if request.method == 'POST':
        # Get the updated data from the form
        email = request.POST.get('email')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        
        # Update the user object
        user = request.user
        user.email = email
        user.first_name = first_name
        user.last_name = last_name
        user.save()
        
        # Add a success message
        messages.success(request, 'Profile updated successfully!')
        
        # Redirect to the profile page to prevent form resubmission
        return redirect('profile')
    
    # For GET requests, just display the profile
    return render(request, 'authentication/profile.html', {'user': request.user})

# History View (requires login)
# Add this function after the profile_view function
@login_required
def history_view(request):
    # Add logic here later to fetch analysis history for the logged-in user
    analysis_history = [] # Placeholder
    return render(request, 'authentication/history.html', {'analysis_history': analysis_history})

@login_required
@require_POST # Ensure this view only accepts POST requests
def delete_history_view(request, analysis_id):
    # Replace AnalysisRecord with your actual history model
    # history_item = get_object_or_404(AnalysisRecord, pk=analysis_id, user=request.user)
    # --- Placeholder ---
    # Since we don't have the model yet, we'll just simulate success
    print(f"Attempting to delete history item {analysis_id} for user {request.user.username}")
    # --- End Placeholder ---

    try:
        # --- Placeholder ---
        # history_item.delete() # Uncomment this when you have the model
        messages.success(request, f"Analysis record #{analysis_id} deleted successfully.")
        # --- End Placeholder ---
    except Exception as e: # Catch potential errors during deletion
         messages.error(request, f"Error deleting analysis record: {e}")

    return redirect('history') # Redirect back to the history page

from django.shortcuts import render, redirect
from django.http import HttpResponse

def send_message_view(request):
    if request.method == 'POST':
        # Process the form data
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')
        
        # Here you can add logic to send an email or save the message to the database
        
        # Redirect to a success page or render a success message
        return HttpResponse("Thank you for your message. We will get back to you soon.")
    else:
        return redirect('contact')


from django.shortcuts import render, redirect

# Replace @login_required with @login_or_register_required for these views

# Replace this:
@login_required
def image_upload(request):
    if request.method == 'POST':
        # Handle image file upload here
        return redirect('image')  # Redirect to image analysis page
    return render(request, 'detection/image_upload.html')

# With this:
@login_required
def image_upload(request):
    if request.method == 'POST':
        # Handle image file upload here
        return redirect('image')  # Redirect to image analysis page
    return render(request, 'detection/image_upload.html')

# Replace this:
@login_required
def video_upload(request):
    if request.method == 'POST':
        inputVideo = request.FILES.get('video')
        if inputVideo:
            # Save the uploaded video
            fs = FileSystemStorage()
            file = fs.save(inputVideo.name, inputVideo)
            fileVideoUrl = fs.path(file)
            fileVideoUrl = os.path.normpath(fileVideoUrl)
            fileVideoUrl = unquote(fileVideoUrl)
            inputVideoUrl = fs.url(file)
            
            # Get video metadata
            try:
                video_metadata = get_video_metadata(fileVideoUrl)
                print("Video metadata:", video_metadata)
            except Exception as e:
                print(f"Error getting video metadata: {e}")
                video_metadata = {}
            
            # Directly analyze the video
            try:
                detection_result = detect_video_forgery(fileVideoUrl)
                print("Detection result:", detection_result)
                
                if detection_result.get('result') in ['Authentic', 'Forged']:
                    result = {
                        'type': detection_result.get('result'),
                        'forged_frames': detection_result.get('f_frames', 0)
                    }
                else:
                    error_message = detection_result.get('message', 'Unknown error during video analysis')
                    return render(request, "detection/video.html", {
                        'error': error_message,
                        'input_video': inputVideoUrl,
                        'video_metadata': video_metadata
                    })
                    
            except Exception as e:
                print(f"Error in video forgery detection: {e}")
                return render(request, "detection/video.html", {
                    'error': f"Error analyzing video: {str(e)}",
                    'input_video': inputVideoUrl,
                    'video_metadata': video_metadata
                })
            
            # Redirect to video analysis page with results
            return render(request, "detection/video.html", {
                'result': result,
                'input_video': inputVideoUrl,
                'video_metadata': video_metadata
            })
            
    return render(request, 'detection/video_upload.html')

def image_analysis(request):
    global fileurl, inputImage, result, infoDict

    # Default context
    context = {
        'input_image': inputImage,
        'result': result,
        'metadata': list(infoDict.items()) if infoDict else None,  # Convert to list
        'ela_url': None,
        'detection_method': "CNN-based Classification with ELA",  # Add default detection method
        'verification_text': "Multiple forensic techniques applied"  # Add default verification text
    }

    # If input image and file path are available, generate ELA
    if inputImage and os.path.exists(fileurl):
        temp_dir = 'temp'
        media_temp_dir = os.path.join(settings.MEDIA_ROOT, temp_dir)
        os.makedirs(media_temp_dir, exist_ok=True)

        output_path = os.path.join(media_temp_dir, 'output.jpg')
        elaImageUrl = f"{settings.MEDIA_URL}{temp_dir}/output.jpg"

        try:
            FID().show_ela(fileurl, save_path=output_path)
            context['ela_url'] = elaImageUrl
        except Exception as e:
            print(f"Error generating ELA on page load: {e}")
            context['error'] = str(e)

    return render(request, 'detection/image.html', context)


def video_analysis(request):
    global fileVideoUrl, inputVideoUrl, result, video_metadata
    
    # Initialize context with available data
    context = {
        'result': result if 'result' in globals() else None,
        'input_video': inputVideoUrl if 'inputVideoUrl' in globals() else None,
        'video_metadata': video_metadata if 'video_metadata' in globals() else {}
    }
    
    return render(request, "detection/video.html", context)

# View for How It Works page
def how_it_works_view(request):
    return render(request, 'pages/how_it_works.html')

# View for FAQs page
def faqs_view(request):
    return render(request, 'pages/faqs.html')

# View for Contact Us page
def contact_view(request):
    if request.method == 'POST':
        # Check if user is authenticated before processing the form
        if not request.user.is_authenticated:
            messages.error(request, "You must be logged in to submit the contact form.")
            return redirect('login')
            
        # Process the form data (e.g., send email)
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')
        print(f"Contact Form Submission:\nName: {name}\nEmail: {email}\nSubject: {subject}\nMessage: {message}")
        # Add success message
        messages.success(request, "Your message has been sent successfully! We'll get back to you soon.")
        return redirect('contact')  # Redirect to clear the form
    # If GET request, just show the empty form
    return render(request, 'pages/contact.html')
