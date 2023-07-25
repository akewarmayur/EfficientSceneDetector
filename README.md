# EfficientSceneDetector
A python package for detecting scenes from a video using audio features. The generated scenes should be with different context.

### Installation
```
pip install -i https://test.pypi.org/simple/ EfficientSceneDetector==0.0.12
```
## Quick Start (Python Code)
```
from EfficientSceneDetector.sceneDetection import SceneDetection
objScene = SceneDetection()
objScene.detect_scenes("input_video.mp4", 4, "True")
```

## Quick Start (Command Line)
```
python main.py -i "input_video.mp4" -w 4 -s "True"
python main.py --input_video "input_video.mp4" ---window 4 --save_scenes "True"
```

#### Details
* input_video.mp4 : your input video path.
* 4 : size of window to be considered while generating scenes. If window is large then large scenes could be generated. If the window size is small then small scenes could be generated
* "True" : If you want to save generated scenes using ffmpeg. Make it "False" if you don't want to generate the scenes.

#### Output
* You will get all the scenes, and a csv file in the "Scenes" folder.
* The csv file columns are : SceneNumber, StartTime (Seconds), StartTime (TimeStamp), EndTime (Seconds), EndTime (TimeStamp) and Path

#### CSV Columns
* SceneNumber : Number of scenes, starting from 1.
* StartTime (Seconds) : Start time of a scene in seconds.
* StartTime (TimeStamp) : Start time of a scene in "%d:%02d:%02d" format.
* EndTime (Seconds) : End time of a scene in seconds.
* EndTime (TimeStamp) : End time of a scene in "%d:%02d:%02d" format.
* Path : Path of generated scenes.

# PIP
```
https://test.pypi.org/project/EfficientSceneDetector/0.0.12/
```

# Help & Contributing
Please submit any bugs/issues or feature requests to the Issue Tracker. Before submission, ensure you search through existing issues (both open and closed) to avoid creating duplicate entries. Pull requests are welcome and encouraged. 

# License
MIT License

Copyright (C) 2023 Mayur Akewar. All rights reserved.
