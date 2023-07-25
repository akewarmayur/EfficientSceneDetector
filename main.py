from EfficientSceneDetector.sceneDetection import SceneDetection
import argparse
import shutil
import os
if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-i', '--input_video', action='store', type=str, required=True)
    my_parser.add_argument('-w', '--window', action='store', type=int, required=True)
    my_parser.add_argument('-s', '--save_scenes', action='store', type=str, required=True)
    args = my_parser.parse_args()
    video_path = args.input_video
    window = args.window
    save_scenes = args.save_scenes
    current_directory = os.getcwd()  # Get the current working directory
    destination_file = os.path.join(current_directory, "tmp.mp4")
    shutil.copy(video_path, destination_file)
    obScene = SceneDetection()
    obScene.detect_scenes("tmp.mp4", window, save_scenes)
# obj = SceneDetection()
#
# obj.detect_scenes("input_video.mp4", 4)