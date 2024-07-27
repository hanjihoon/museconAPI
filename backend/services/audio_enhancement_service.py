import logging
from pydub import AudioSegment
import numpy as np
from scipy import signal
from pydub.effects import (
    normalize,
    compress_dynamic_range,
    high_pass_filter,
    low_pass_filter,
)
from pydub.silence import split_on_silence

class AudioEnhancementService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def enhance_audio(self, audio_segment):
        # AudioSegment를 numpy 배열로 변환
        audio_data = np.array(audio_segment.get_array_of_samples())
        
        # 스테레오인 경우 처리
        if audio_segment.channels == 2:
            audio_data = audio_data.reshape((-1, 2))
        
        # 노이즈 감소 적용
        enhanced_data = self.reduce_noise(audio_data, audio_segment.frame_rate)
        
        # numpy 배열을 다시 AudioSegment로 변환
        enhanced_audio = AudioSegment(
            enhanced_data.tobytes(),
            frame_rate=audio_segment.frame_rate,
            sample_width=audio_segment.sample_width,
            channels=audio_segment.channels
        )
        
        return enhanced_audio

    def _split_and_process(self, audio):
        """오디오를 무음 구간으로 분할하고 각 구간을 처리합니다.
        
        Args:
            audio (AudioSegment): 처리할 오디오
        
        Returns:
            AudioSegment: 처리된 오디오
        """
        chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
        enhanced_chunks = []
        for chunk in chunks:
            chunk = normalize(chunk)             # Normalize
            chunk = compress_dynamic_range(chunk) # Dynamic range compression
            chunk = high_pass_filter(chunk, 80)  # High-pass filter (remove low-frequency noise)
            chunk = low_pass_filter(chunk, 12000) # Low-pass filter (remove high-frequency noise)
            enhanced_chunks.append(chunk)
        return sum(enhanced_chunks) # Combine processed chunks

    def reduce_noise(self, audio_data, sr):
        # 저역 통과 필터 적용
        nyq = 0.5 * sr
        cutoff = 10000  # 차단 주파수 (Hz)
        b, a = signal.butter(6, cutoff / nyq, btype='low', analog=False)
        filtered_audio = signal.lfilter(b, a, audio_data)
        return filtered_audio