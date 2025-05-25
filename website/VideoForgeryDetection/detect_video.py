import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from keras.models import load_model

# Define the project root directory dynamically
# Goes up three levels from the current file's directory (VideoForgeryDetection -> website -> ForgeryCheck)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def detect_video_forgery(vid_src):
    vid = []
    sumFrames = 0
    cap = cv2.VideoCapture(vid_src)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file: {vid_src}")
        return {'result': 'Error', 'f_frames': 0, 'message': 'Could not open video file.'}

    fps = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            fps = cap.get(cv2.CAP_PROP_FPS)
            break
        try:
            # Add error handling for resize operation
            b = cv2.resize(frame,(320,240),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            sumFrames +=1
            vid.append(b)
        except cv2.error as e:
            print(f"Error resizing frame: {e}")
            # Optionally continue to next frame or break
            continue # Skip this frame
        except Exception as e:
            print(f"An unexpected error occurred while processing a frame: {e}")
            continue # Skip this frame

    cap.release()

    print("\nNo. Of Frames processed in the Video: ",sumFrames)

    # --- Add check for empty video ---
    if sumFrames == 0:
        print("\nError: No frames were processed from the video. Cannot predict.")
        return {'result': 'Error', 'f_frames': 0, 'message': 'No frames processed from video.'}
    # --- End of added check ---

    Xtest = np.array(vid)

    print("\nPredicting !! ")
    # Construct the path relative to the calculated project root
    model_path = os.path.join(PROJECT_ROOT, 'ml_models', 'forgery_model_me.hdf5') # Corrected path construction

    try:
        # model = load_model('C://Users//User//ML//Video_Forgery_Detection//ResNet50_Model//forgery_model_me.hdf5') # Old path
        model = load_model(model_path) # Updated path
        output = model.predict(Xtest)
    except ValueError as e:
        # Catch potential errors during prediction (though the check above should prevent the math domain error)
        print(f"Error during model prediction: {e}")
        return {'result': 'Error', 'f_frames': 0, 'message': f'Prediction error: {e}'}
    except Exception as e:
        print(f"An unexpected error occurred during model loading or prediction: {e}")
        return {'result': 'Error', 'f_frames': 0, 'message': f'Model error: {e}'}


    output = output.reshape((-1))
    results = []
    for i in output:
        if i>0.5:
            results.append(1)
        else:
            results.append(0)


    no_of_forged = sum(results)

    print('No of forged----no_of_forged:',no_of_forged)
            
    if no_of_forged <=0:
        print("\nThe video is not forged")
        return {'result':'Authentic','f_frames':0}
        
    else:
        print("\nThe video is forged")
        print("\nNumber of Forged Frames in the video: ",no_of_forged)
        return {'result':'Forged','f_frames':no_of_forged}
