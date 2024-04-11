import os
import sys

os.system('git clone https://github.com/facebookresearch/av_hubert.git')
os.chdir('/home/user/app/av_hubert')
os.system('git submodule init')
os.system('git submodule update')
os.chdir('/home/user/app/av_hubert/fairseq')
os.system('pip install ./')
os.system('pip install scipy')
os.system('pip install sentencepiece')
os.system('pip install python_speech_features')
os.system('pip install scikit-video')
os.system('pip install transformers')
os.system('pip install gradio==3.12')
os.system('pip install numpy==1.23.3')


# sys.path.append('/home/user/app/av_hubert')
sys.path.append('/home/user/app/av_hubert/avhubert')

print(sys.path)
print(os.listdir())
print(sys.argv, type(sys.argv))
sys.argv.append('dummy')



import dlib, cv2, os
import numpy as np
import skvideo
import skvideo.io
from tqdm import tqdm
from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
from base64 import b64encode
import torch
import cv2
import tempfile
from argparse import Namespace
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig
from huggingface_hub import hf_hub_download
import gradio as gr
from pytube import YouTube

# os.chdir('/home/user/app/av_hubert/avhubert')

user_dir = "/home/user/app/av_hubert/avhubert"
utils.import_user_module(Namespace(user_dir=user_dir))
data_dir = "/home/user/app/video"

ckpt_path = hf_hub_download('vumichien/AV-HuBERT', 'model.pt')
face_detector_path = "/home/user/app/mmod_human_face_detector.dat"
face_predictor_path = "/home/user/app/shape_predictor_68_face_landmarks.dat"
mean_face_path = "/home/user/app/20words_mean_face.npy"
mouth_roi_path = "/home/user/app/roi.mp4"
modalities = ["video"]
gen_subset = "test"
gen_cfg = GenerationConfig(beam=20)
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
models = [model.eval().cuda() if torch.cuda.is_available() else model.eval() for model in models]
saved_cfg.task.modalities = modalities
saved_cfg.task.data = data_dir
saved_cfg.task.label_dir = data_dir
task = tasks.setup_task(saved_cfg.task)
generator = task.build_generator(models, gen_cfg)

def get_youtube(video_url):
    yt = YouTube(video_url)
    abs_video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    print("Success download video")
    print(abs_video_path)
    return abs_video_path
    
def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_locations  = detector(gray, 1)
    coords = None
    for (_, face_location) in enumerate(face_locations):
        if torch.cuda.is_available():
            rect = face_location.rect
        else:
            rect = face_location
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def preprocess_video(input_video_path):
    if torch.cuda.is_available():
        detector = dlib.cnn_face_detection_model_v1(face_detector_path)
    else:
        detector = dlib.get_frontal_face_detector()
    
    predictor = dlib.shape_predictor(face_predictor_path)
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]
    videogen = skvideo.io.vread(input_video_path)
    frames = np.array([frame for frame in videogen])
    landmarks = []
    for frame in tqdm(frames):
        landmark = detect_landmark(frame, detector, predictor)
        landmarks.append(landmark)
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                          window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    write_video_ffmpeg(rois, mouth_roi_path, "/usr/bin/ffmpeg")
    return mouth_roi_path

def predict(process_video):
    num_frames = int(cv2.VideoCapture(process_video).get(cv2.CAP_PROP_FRAME_COUNT))

    tsv_cont = ["/\n", f"test-0\t{process_video}\t{None}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
    label_cont = ["DUMMY\n"]
    with open(f"{data_dir}/test.tsv", "w") as fo:
      fo.write("".join(tsv_cont))
    with open(f"{data_dir}/test.wrd", "w") as fo:
      fo.write("".join(label_cont))
    task.load_dataset(gen_subset, task_cfg=saved_cfg.task)

    def decode_fn(x):
        dictionary = task.target_dictionary
        symbols_ignore = generator.symbols_to_strip_from_output
        symbols_ignore.add(dictionary.pad())
        return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

    itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
    sample = next(itr)
    if torch.cuda.is_available():
        sample = utils.move_to_cuda(sample)
    hypos = task.inference_step(generator, models, sample)
    ref = decode_fn(sample['target'][0].int().cpu())
    hypo = hypos[0][0]['tokens'].int().cpu()
    hypo = decode_fn(hypo)
    return hypo


# ---- Gradio Layout -----
youtube_url_in = gr.Textbox(label="Youtube url", lines=1, interactive=True)
video_in = gr.Video(label="Input Video", mirror_webcam=False, interactive=True)
video_out = gr.Video(label="Audio Visual Video", mirror_webcam=False, interactive=True) 
demo = gr.Blocks()
demo.encrypt = False
text_output = gr.Textbox()

with demo:
    gr.Markdown('''
            <div>
            <h1 style='text-align: center'>Speech Recognition from Visual Lip Movement by Audio-Visual Hidden Unit BERT Model (AV-HuBERT)</h1>
            This space uses AV-HuBERT models from <a href='https://github.com/facebookresearch' target='_blank'><b>Meta Research</b></a> to recoginze the speech from Lip Movement
            <figure>
              <img src="https://huggingface.co/vumichien/AV-HuBERT/resolve/main/lipreading.gif" alt="Audio-Visual Speech Recognition">
              <figcaption> Speech Recognition from visual lip movement
              </figcaption>
            </figure>
            </div>
        ''')
    with gr.Row():
            gr.Markdown('''
            ### Reading Lip movement with youtube link using Avhubert
            ##### Step 1a. Download video from youtube (Note: the length of video should be less than 10 seconds if not it will be cut and the face should be stable for better result)
            ##### Step 1b. You also can upload video directly 
            ##### Step 2. Generating landmarks surrounding mouth area
            ##### Step 3. Reading lip movement.
            ''')
    with gr.Row():         
        gr.Markdown('''
            ### You can test by following examples:
            ''')
    examples = gr.Examples(examples=
            [ "https://www.youtube.com/watch?v=ZXVDnuepW2s", 
              "https://www.youtube.com/watch?v=X8_glJn1B8o", 
              "https://www.youtube.com/watch?v=80yqL2KzBVw"],
          label="Examples", inputs=[youtube_url_in])
    with gr.Column():
          youtube_url_in.render()
          download_youtube_btn = gr.Button("Download Youtube video")
          download_youtube_btn.click(get_youtube, [youtube_url_in], [
              video_in])
          print(video_in)
    with gr.Row():  
        video_in.render()
        video_out.render()
    with gr.Row():
        detect_landmark_btn = gr.Button("Detect landmark")
        detect_landmark_btn.click(preprocess_video, [video_in], [
            video_out])
        predict_btn = gr.Button("Predict")
        predict_btn.click(predict, [video_out], [
            text_output])
    with gr.Row():
        # video_lip = gr.Video(label="Audio Visual Video", mirror_webcam=False) 
        text_output.render()

        
        
demo.launch(debug=True)