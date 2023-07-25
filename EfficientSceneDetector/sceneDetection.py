import pandas as pd
import shutil
import os
import soundfile as sf
import resampy
import numpy as np
import tensorflow as tf
import re
from EfficientSceneDetector.helperFiles.utils import Utils
import EfficientSceneDetector.helperFiles.params as yamnet_params
import EfficientSceneDetector.helperFiles.yam as yamnet_model


class SceneDetection:

    def load_model(self):
        params = yamnet_params.Params()
        yamnet = yamnet_model.yamnet_frames_model(params)
        yamnet.load_weights('EfficientSceneDetector/helperFiles/model.h5')
        yamnet_classes = yamnet_model.class_names('EfficientSceneDetector/helperFiles/yam_class_map.csv')
        return params, yamnet, yamnet_classes

    def AudioClassifier(self, wav_file_path, params, yamnet, yamnet_classes):
        wav_data, sr = sf.read(wav_file_path, dtype=np.int16)
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        waveform = wav_data / tf.int16.max  # 32768.0  # Convert to [-1.0, +1.0]
        waveform = waveform.astype('float32')

        # Convert to mono and the sample rate expected by YAMNet.
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        if sr != params.sample_rate:
            waveform = resampy.resample(waveform, sr, params.sample_rate)

        # Predict YAMNet classes.
        scores, embeddings, spectrogram = yamnet(waveform)
        prediction = np.mean(scores, axis=0)
        # Report the highest-scoring classes and their scores.
        top5_i = np.argsort(prediction)[::-1][:5]
        classes = []
        predictions = []
        for i in top5_i:
            classes.append(yamnet_classes[i])
            predictions.append(prediction[i])

        return classes, predictions

    def list_of_clips(self, video_path):
        objUtils = Utils()
        audioFiles = objUtils.slice_video_convert2audio(video_path)
        return audioFiles

    def classify_audio(self, video_path):
        audioFiles = self.list_of_clips(video_path)
        df = pd.DataFrame(columns=["FileName", "StartTime", "EndTime", "Top3Predictions",
                                   "Score"], )
        params, yamnet, yamnet_classes = self.load_model()
        for audio_file in audioFiles:
            tmp = audio_file.split("/")
            name = tmp[len(tmp) - 1]
            numbers = re.findall(r'\d+', name)
            bb = [int(num) for num in numbers]
            StartTime = bb[0]
            EndTime = bb[1]
            classes, predictions = self.AudioClassifier(audio_file, params, yamnet, yamnet_classes)
            predictions = [str(i) for i in predictions]
            tm = [name, StartTime, EndTime, ":".join(classes[:4]), ":".join(predictions[:4])]
            df_len = len(df)
            df.loc[df_len] = tm
        return df

    def detect_scenes(self, input_video, window, save_scenes):
        objUtils = Utils()
        try:
            df = self.classify_audio(input_video)
            ee = df["Top3Predictions"].to_list()
            prediction_list = [i.split(":")[0] for i in ee]
            start_time_list = df["StartTime"].to_list()
            end_time_list = df["EndTime"].to_list()
            prediction_list = objUtils.process_list_with_window_size(prediction_list, window)
            grouped_list = objUtils.group_similar_elements(prediction_list)
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
                if save_scenes == "True" or save_scenes == "TRUE":
                    objUtils.cut_video(input_video, "Scenes/" + str(scene) + ".mp4", value[0],
                                       value[1])
                tmm = [scene, value[0], objUtils.convert2timestamp(value[0]), value[1],
                       objUtils.convert2timestamp(value[1]),
                       "Scenes/" + str(scene) + ".mp4"]
                lengthDF = len(scenesDF)
                scenesDF.loc[lengthDF] = tmm

            scenesDF.to_csv("Scenes/scenes.csv")
            try:
                shutil.rmtree("wavefiles")
            except:
                pass

            # try:
            #     os.remove("tmp.mp4")
            # except:
            #     pass
        except Exception as e:
            print("Exception in detecting scenes:", e)



