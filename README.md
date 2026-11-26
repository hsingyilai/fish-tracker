# Fish tracker

Track a pet fish on camera and, later, flag changes from its normal behavior.

**Status:** data collection. No model has been trained yet.

## Setup (PowerShell)

    py -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt

## Collect data

    .venv\Scripts\python record.py --preview
    .venv\Scripts\python record.py --clips 12 --every 3600
    .venv\Scripts\python extract_frames.py --interval 2

Videos from a phone can be copied into `data/videos/` instead.

## Plan

1. Record clips across several days and lighting conditions.
2. Label the fish in a few hundred frames.
3. Fine-tune a small detector; test it on days it was not trained on.
4. Track position and speed.
5. Alert on changes from normal behavior.
