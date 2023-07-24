import re
import pandas as pd
from utils import Utils
import modelFiles.params as yamnet_params
import modelFiles.yamnet as yamnet_model
import soundfile as sf
import resampy
import numpy as np
import tensorflow as tf


class ClassifyAudio:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.slice_per_sec = 5
        self.objUtils = Utils()
        self.model_path = 'modelFiles/yamnet.h5'
        self.yamnet_class_map = 'modelFiles/yamnet_class_map.csv'

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def load_yamnet_model(self):
        params = yamnet_params.Params()
        yamnet = yamnet_model.yamnet_frames_model(params)
        yamnet.load_weights(self.model_path)
        yamnet_classes = yamnet_model.class_names(self.yamnet_class_map)

        return params, yamnet, yamnet_classes

    def yamnetClassifier(self, wav_file_path, params, yamnet, yamnet_classes):
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
        audioFiles = self.objUtils.slice_video_convert2audio(video_path)
        return audioFiles

    def classify_audio(self, video_path):
        audioFiles = self.list_of_clips(video_path)
        df = pd.DataFrame(columns=["FileName", "StartTime", "EndTime", "Top3Predictions",
                                   "Score"], )
        params, yamnet, yamnet_classes = self.load_yamnet_model()
        for audio_file in audioFiles:
            tmp = audio_file.split("/")
            name = tmp[len(tmp) - 1]
            numbers = re.findall(r'\d+', name)
            bb = [int(num) for num in numbers]
            StartTime = bb[0]
            EndTime = bb[1]
            classes, predictions = self. yamnetClassifier(audio_file, params, yamnet, yamnet_classes)
            predictions = [str(i) for i in predictions]
            tm = [name, StartTime, EndTime, ":".join(classes[:4]), ":".join(predictions[:4])]
            df_len = len(df)
            df.loc[df_len] = tm
        return df

# obj = ClassifyAudio()
# df = obj.classify_audio("input_video.mp4")
# df.to_csv("tt.csv")