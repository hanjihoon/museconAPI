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
    
    def spectral_subtraction(audio_data, sr, frame_size=2048, hop_size=512):
        # 스펙트로그램 계산
        f, t, Sxx = signal.spectrogram(audio_data, sr, nperseg=frame_size, noverlap=frame_size-hop_size)
        
        # 노이즈 프로파일 추정 (첫 몇 프레임을 노이즈로 가정)
        noise_profile = np.mean(Sxx[:, :10], axis=1)
        
        # 스펙트럼 서브트랙션
        Sxx_clean = np.maximum(Sxx - noise_profile[:, np.newaxis], 0)
        
        # 역변환
        _, enhanced_audio = signal.istft(Sxx_clean, sr, nperseg=frame_size, noverlap=frame_size-hop_size)
        
        return enhanced_audio
    
    def spectral_subtraction(audio_data, sr, frame_size=2048, hop_size=512):
        # 스펙트로그램 계산
        f, t, Sxx = signal.spectrogram(audio_data, sr, nperseg=frame_size, noverlap=frame_size-hop_size)
        
        # 노이즈 프로파일 추정 (첫 몇 프레임을 노이즈로 가정)
        noise_profile = np.mean(Sxx[:, :10], axis=1)
        
        # 스펙트럼 서브트랙션
        Sxx_clean = np.maximum(Sxx - noise_profile[:, np.newaxis], 0)
        
        # 역변환
        _, enhanced_audio = signal.istft(Sxx_clean, sr, nperseg=frame_size, noverlap=frame_size-hop_size)
        
        return enhanced_audio
    
    def harmonic_enhancement(audio_data, sr, frame_size=2048, hop_size=512):
        f, t, Sxx = signal.spectrogram(audio_data, sr, nperseg=frame_size, noverlap=frame_size-hop_size)
        
        # 하모닉 구조 강화
        harmonic_enhanced = np.copy(Sxx)
        for i in range(1, 5):  # 4개의 하모닉까지 강화
            harmonic_enhanced += np.roll(Sxx, int(i * len(f) / 440), axis=0) * (0.5 ** i)
        
        _, enhanced_audio = signal.istft(harmonic_enhanced, sr, nperseg=frame_size, noverlap=frame_size-hop_size)
        return enhanced_audio
    
    def multiband_compression(audio_data, sr):
        # 다중 대역으로 분할
        b1, a1 = signal.butter(4, 200 / (sr/2), btype='lowpass')
        b2, a2 = signal.butter(4, [200 / (sr/2), 2000 / (sr/2)], btype='bandpass')
        b3, a3 = signal.butter(4, 2000 / (sr/2), btype='highpass')
        
        low = signal.lfilter(b1, a1, audio_data)
        mid = signal.lfilter(b2, a2, audio_data)
        high = signal.lfilter(b3, a3, audio_data)
        
        # 각 대역별 압축 적용
        compressed_low = compress_dynamic_range(AudioSegment(low.tobytes(), frame_rate=sr, sample_width=2, channels=1))
        compressed_mid = compress_dynamic_range(AudioSegment(mid.tobytes(), frame_rate=sr, sample_width=2, channels=1))
        compressed_high = compress_dynamic_range(AudioSegment(high.tobytes(), frame_rate=sr, sample_width=2, channels=1))
        
        # 압축된 대역 결합
        return compressed_low.overlay(compressed_mid).overlay(compressed_high)
    
    