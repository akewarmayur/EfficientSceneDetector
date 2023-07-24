from audioClassification import ClassifyAudio
import pandas as pd
import shutil
import os
import argparse
from utils import Utils


class SceneDetection:

    def __init__(self):
        self.objAC = ClassifyAudio()
        self.objUtils = Utils()

    def detect_scenes(self, input_video, window, save_scenes=True):
        try:
            df = self.objAC.classify_audio(input_video)
            ee = df["Top3Predictions"].to_list()
            prediction_list = [i.split(":")[0] for i in ee]
            start_time_list = df["StartTime"].to_list()
            end_time_list = df["EndTime"].to_list()
            prediction_list = self.objUtils.process_list_with_window_size(prediction_list, window)
            grouped_list = self.objUtils.group_similar_elements(prediction_list)
            if not os.path.exists("Scenes/"):
                os.makedirs("Scenes/")
            scenes = {}
            scene = 1
            for h in grouped_list:
                start = h[0]
                end = h[len(h) - 1]
                scenes[scene] = [start_time_list[start], end_time_list[end]]
                scene += 1
            scenesDF = pd.DataFrame(
                columns=["SceneNumber", "StartTime (Seconds)", "StartTime (TimeStamp)", "EndTime (Seconds)",
                         "EndTime (TimeStamp)", "Path"])
            for scene, value in scenes.items():
                if save_scenes:
                    self.objUtils.cut_video(input_video, "Scenes/" + str(scene) + ".mp4", value[0], value[1])
                tmm = [scene, value[0], self.objUtils.convert2timestamp(value[0]), value[1],
                       self.objUtils.convert2timestamp(value[1]),
                       "Scenes/" + str(scene) + ".mp4"]
                lengthDF = len(scenesDF)
                scenesDF.loc[lengthDF] = tmm

            scenesDF.to_csv("Scenes/scenes.csv")
            try:
                shutil.rmtree("wavefiles")
            except:
                pass

            try:
                os.remove("tmp.mp4")
            except:
                pass
        except Exception as e:
            print("Exception in detecting scenes:", e)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-i', '--input_video', action='store', type=str, required=True)
    my_parser.add_argument('-w', '--window', action='store', type=int, required=True)
    my_parser.add_argument('-s', '--save_scenes', action='store', type=bool, required=False)
    args = my_parser.parse_args()
    video_path = args.input_video
    window = args.window
    save_scenes = args.save_scenes
    current_directory = os.getcwd()  # Get the current working directory
    destination_file = os.path.join(current_directory, "tmp.mp4")
    shutil.copy(video_path, destination_file)
    obScene = SceneDetection()
    obScene.detect_scenes("tmp.mp4", window, save_scenes)
