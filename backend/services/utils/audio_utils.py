from pydub import AudioSegment

def load_audio_file(file_path):
    """
    오디오 파일을 로드합니다.

    Args:
        file_path (str): 오디오 파일 경로

    Returns:
        AudioSegment: 로드된 오디오 세그먼트
    """
    return AudioSegment.from_file(file_path)

def change_tempo(audio, speed_factor):
    """
    오디오의 템포를 변경합니다.

    Args:
        audio (AudioSegment): 원본 오디오
        speed_factor (float): 속도 변경 비율 (1.0 = 원래 속도)

    Returns:
        AudioSegment: 템포가 변경된 오디오
    """
    return audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed_factor)
    }).set_frame_rate(audio.frame_rate)

def normalize_audio(audio, target_dBFS=-20.0):
    """
    오디오 볼륨을 정규화합니다.

    Args:
        audio (AudioSegment): 원본 오디오
        target_dBFS (float): 목표 dBFS 값

    Returns:
        AudioSegment: 정규화된 오디오
    """
    change_in_dBFS = target_dBFS - audio.dBFS
    return audio.apply_gain(change_in_dBFS)