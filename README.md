# Pitching Analysis Tools
You will need python installed to run this program.

## Start up guide
First download this repo if you haven't already:

`git clone https://github.com/cchuter/pitcher.git`

Next, setup your python environment:
```
cd pitcher
python -m venv .
source bin/activate
pip install mediapipe opencv-python numpy matplotlib
```

# Running the tools
Make sure you have a video in *.mov or *.mp4 format of a pitcher you want to analyze. Name it `pitcher_video.mov` in this directory.

Then run:

`python pitcher_analysis.py`

You should see a video come up mapping the pitcher's joints and torso (and possibly their face). Then a ploy of arm and leg angles will save and dispay. An example of Luke Weaver pitching is analysed from a Tik Tok video.

Two files will be created:

```
pitcher_video_analyzed.mov
pitcher_joint_angles.png
```

The first file is the joint angles determined by frame and the second is a 2D plot over time.



