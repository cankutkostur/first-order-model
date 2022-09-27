import os

import cv2
import moviepy.editor as mpe
import numpy as np
import ffmpeg


def _get_dimensions(img, face_position):
    h, w, _ = img.shape
    face_w = int(w * face_position[2])
    face_h = int(h * face_position[3])

    start_w = max(0, int(w * face_position[0] - face_w / 2))
    start_h = max(0, int(h * face_position[1] - face_h / 2))

    end_w = min(start_w + face_w, w)
    end_h = min(start_h + face_h, h)

    edge = min((end_h - start_h), (end_w - start_w))

    return start_w, start_h, edge


def extract_face(img, face_position):
    start_w, start_h, edge = _get_dimensions(img, face_position)

    face = img[start_h:start_h+edge, start_w:start_w+edge]
    return face


def put_face_to_frames(source_image, frames, face_position, smooth_edges=False):
    start_w, start_h, edge = _get_dimensions(source_image, face_position)

    # the mask for pasting restored faces back
    mask = np.zeros((512, 512), np.float32)
    if smooth_edges:
        cv2.rectangle(mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = cv2.resize(mask, (edge, edge))

    for i in range(len(frames)):
        whole = source_image.copy()
        face = cv2.resize(frames[i], (edge, edge), cv2.INTER_CUBIC)
        if smooth_edges:
            whole[start_h:start_h+edge, start_w:start_w+edge, :] = whole[start_h:start_h+edge, start_w:start_w+edge, :] * (1-mask)
            whole[start_h:start_h+edge, start_w:start_w+edge, :] = whole[start_h:start_h+edge, start_w:start_w+edge, :] + face[:, :, :] * mask
        else:
            whole[start_h:start_h + edge, start_w:start_w + edge, :] = face[:, :, :]
        frames[i] = whole


def resize_if_needed(image, face_position, limit):
    max_size = limit
    _, _, edge = _get_dimensions(image, face_position)
    h, w, _ = image.shape
    image_max_size = max(h, w)
    if edge > 512:
        percent = 512 / edge
        new_size = int(image_max_size * percent)
        max_size = min(new_size, max_size)

    image = downscale(image, max_size)

    return image


def downscale(im, size):
    if im.shape[0] > size or im.shape[1] > size:
        scale_percent = size / max(im.shape[0], im.shape[1])
        width = int(im.shape[1] * scale_percent)
        height = int(im.shape[0] * scale_percent)
        new_size = (16 * (width // 16), 16 * (height // 16))
        im = cv2.resize(im, new_size)

    return im


def add_sound(video_path, sound_path):
    # audio = ffmpeg.input(sound_path)
    # video = ffmpeg.input(video_path)
    # ffmpeg.concat(video, audio, v=1, a=1).output(video_path).run(overwrite_output=True)

    my_clip = mpe.VideoFileClip(video_path)
    audio_background = mpe.AudioFileClip(sound_path)
    # final_audio = mpe.CompositeAudioClip([my_clip.audio, audio_background])
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile("temp.mp4", audio=True, audio_codec="aac")
    os.remove(video_path)
    os.rename("temp.mp4", video_path)


