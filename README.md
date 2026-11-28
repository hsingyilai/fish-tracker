# Fish tracker

Track a pet fish on camera and, later, flag changes from its normal behavior.

**Status:** the pipeline runs from recording to training, and a live view shows
the detections. The latest detector was trained on frames from eight one-minute
clips and checked on two others, all recorded on the same day with the camera in
one position, so its accuracy on other days or setups is untested. Tracking and
behavior alerts are not built yet.

## Setup (PowerShell)

    py -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt

## Collect data

    .venv\Scripts\python record.py --preview
    .venv\Scripts\python record.py --clips 12 --every 3600
    .venv\Scripts\python extract_frames.py --interval 2

Videos from a phone can be copied into `data/videos/` instead.

If the camera fails, `record.py` retries. Footage from before the failure is
kept, so one clip may be split into several files.

Once its images are saved, a video is moved to `data/videos/extracted/`, so it
is not processed again. To extract it again, for example with a new interval,
move it back into `data/videos/`.

## Label and train

Training needs an NVIDIA GPU. These versions are for Python 3.14 on Windows:

    .venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
    .venv\Scripts\python -m pip install ultralytics==8.4.220

Then label, and train on the videos you name (full names, or how they start):

    .venv\Scripts\python label.py
    .venv\Scripts\python train.py 20261127_09 20261127_1

Every labeled image of those videos is used. Nothing is held out for
validation, so the scores printed during training say nothing about new footage.

Ultralytics is licensed under AGPL-3.0.

## Score the model

Name videos the model was not trained on:

    .venv\Scripts\python evaluate.py 20261126

It refuses videos used in training. Add `--save-mistakes` to keep pictures of
what the model got wrong under `data/runs/eval/`.

## Watch live

Show the camera with the newest model's box drawn on it (press q to quit):

    .venv\Scripts\python live.py

Nothing is saved. Close `record.py` and other camera apps first.

## Plan

1. Record clips across several days and lighting conditions.
2. Label the fish in a few hundred frames.
3. Fine-tune a small detector; test it on days it was not trained on.
4. Track position and speed.
5. Alert on changes from normal behavior.
