import re
import os
import moviepy.editor as mp
import math
import ffmpeg
from itertools import groupby


class Utils:

    def __init__(self):
        self.slice_per_sec = 2

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def convertVideoToWavSlice(self, videoFilePath, saveWavFiles, slicePerSec):
        print("videoFilePath", videoFilePath)
        videoClip = mp.VideoFileClip(videoFilePath)
        videoDuration = videoClip.duration
        iterCount = int(math.ceil(videoDuration) / int(slicePerSec))
        start = 0
        end = slicePerSec
        audioFiles = []
        for idx in range(iterCount):
            videoSubclip = videoClip.subclip(start, end)
            audioclip = videoSubclip.audio
            try:
                audioFileName = saveWavFiles + videoFilePath.split("/")[-1].split(".")[0] + "_" + str(
                    start) + "_" + str(
                    end) + ".wav"
            except:
                audioFileName = saveWavFiles + videoFilePath.split("\\")[-1].split(".")[0] + "_" + str(
                    start) + "_" + str(
                    end) + ".wav"
            try:
                audioclip.write_audiofile(audioFileName)
                audioFiles.append(audioFileName)
            except Exception as e:
                continue
            start = end
            end = end + slicePerSec
            if (end != videoDuration) and (end > videoDuration):
                end = videoDuration
        audioFiles.sort(key=self.natural_keys)
        return audioFiles

    def slice_video_convert2audio(self, video_path):
        try:
            saveWavFiles = "wavefiles/"
            if not os.path.exists(saveWavFiles):
                os.makedirs(saveWavFiles)
            audioFiles = self.convertVideoToWavSlice(video_path, saveWavFiles, self.slice_per_sec)
            return audioFiles
        except Exception as e:
            print("Exception in slice video:", e)
            return []

    def cut_video(self, input_video_file, output_video_file, start_time, end_time):
        ffmpeg.input(input_video_file, ss=start_time).output(output_video_file, to=end_time - start_time).run()

    def convert2timestamp(self, seconds):
        seconds = seconds % (24 * 3600)
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return "%d:%02d:%02d" % (hour, minutes, seconds)

    def process_list_with_window_size(self, lst, window_size):
        res = []
        grouped_list = [list(group) for key, group in groupby(lst)]
        for ss, i in enumerate(grouped_list):
            if ss == 0:
                for a in i:
                    res.append(a)
            else:
                if len(i) <= window_size:
                    for a in range(len(i)):
                        res.append(res[len(res) - 1])
                elif len(i) > window_size and res[len(res) - 1] != i[0]:
                    for a in i:
                        res.append(a)
                else:
                    for a in range(len(i)):
                        res.append(res[len(res) - 1])
        return res

    def group_similar_elements(self, lst):
        grouped_list = []
        for key, group in groupby(enumerate(lst), lambda x: x[1]):
            indices = [index for index, _ in group]
            grouped_list.append(indices)
        return grouped_list
