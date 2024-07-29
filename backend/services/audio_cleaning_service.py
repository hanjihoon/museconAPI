import logging
import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, lfilter
import pywt

class AudioCleaningService:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def clean_audio(self, input_file, output_file):
        """
        입력 WAV 파일에 오디오 처리 기술을 적용하여 노이즈를 제거하고 음질을 개선합니다.
        """
        # 오디오 로드
        audio = AudioSegment.from_wav(input_file)
        original_length_ms = len(audio)
        
        # 노이즈 게이트 적용
        gated_audio = self.apply_noise_gate(audio)
        
        # AudioSegment를 numpy 배열로 변환
        samples = np.array(gated_audio.get_array_of_samples())
        
        # 로우패스 필터 적용
        filtered_data = self.apply_lowpass_filter(samples, cutoff=2000, fs=gated_audio.frame_rate)
        
        # 웨이블릿 디노이징
        denoised_data = self.wavelet_denoising(filtered_data)
        
        # numpy 배열을 다시 AudioSegment로 변환
        cleaned_audio = AudioSegment(
            denoised_data.tobytes(),
            frame_rate=gated_audio.frame_rate,
            sample_width=gated_audio.sample_width,
            channels=gated_audio.channels
        )
        
        # 원본 길이로 조정
        if len(cleaned_audio) < original_length_ms:
            cleaned_audio = cleaned_audio + AudioSegment.silent(duration=original_length_ms - len(cleaned_audio))
        else:
            cleaned_audio = cleaned_audio[:original_length_ms]
        
        # 결과 저장
        cleaned_audio.export(output_file, format="wav")
        
        self.logger.info(f"원본 오디오 길이: {original_length_ms}ms, 처리된 오디오 길이: {len(cleaned_audio)}ms")

    def apply_noise_gate(self, audio, threshold_db=-50, chunk_length_ms=50):
        """
        오디오에 노이즈 게이트를 적용하여 배경 노이즈를 제거합니다.

        이 함수는 입력 오디오를 작은 청크로 나누고, 각 청크의 볼륨이 지정된 임계값 이하인 경우
        해당 청크를 무음으로 대체합니다.

        Args:
            audio (AudioSegment): 처리할 오디오 세그먼트
            threshold_db (float): 노이즈 게이트 임계값 (데시벨). 기본값은 -40dB
            chunk_length_ms (int): 각 오디오 청크의 길이 (밀리초). 기본값은 50ms

        Returns:
            AudioSegment: 노이즈 게이트가 적용된 오디오 세그먼트

        주의:
            - 이 함수는 입력 오디오를 모노로 변환합니다.
            - 매우 짧은 chunk_length_ms는 처리 시간을 증가시킬 수 있습니다.
        """
        def gate(chunk):
            if chunk.dBFS < threshold_db:
                return AudioSegment.silent(duration=len(chunk))
            return chunk

        # 오디오를 짧은 청크로 나누고 각 청크에 게이트 적용
        chunk_length = 50  # 밀리초
        gated_chunks = []
        for i in range(0, len(audio), chunk_length):
            chunk = audio[i:i+chunk_length]
            gated_chunk = gate(chunk)
            gated_chunks.append(gated_chunk)

        # 게이트된 청크들을 다시 합치기
        gated_audio = sum(gated_chunks)

        return gated_audio

    def apply_lowpass_filter(self, data, cutoff, fs, order=5):
        """
        로우패스 필터를 적용하여 고주파 노이즈를 제거합니다.
        """
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def wavelet_denoising(self, data, wavelet='db4', level=1):
        """
        웨이블릿 변환을 사용하여 신호에서 노이즈를 제거합니다.

        Args:
            data (np.array): 입력 오디오 데이터
            wavelet (str): 사용할 웨이블릿 유형. 기본값은 'db4'
            level (int): 분해 레벨. 기본값은 1

        Returns:
            np.array: 디노이징된 오디오 데이터
        """
        # 데이터 정규화
        data = data.astype(np.float64)  # 데이터 타입을 float64로 변경
        data_max = np.max(np.abs(data))
        if data_max > 0:
            data = data / data_max

        coeff = pywt.wavedec(data, wavelet, mode="per", level=level)
        sigma = (1/0.6745) * np.median(np.abs(coeff[-level]))
        uthresh = sigma * np.sqrt(2 * np.log(len(data)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
        
        # 역변환 및 원래 스케일로 복원
        denoised = pywt.waverec(coeff, wavelet, mode="per")
        denoised = denoised * data_max
        return denoised.astype(np.int16)  # 다시 int16으로 변환

    def compare_audio(self, original_file, processed_file, output_file):
        """
        원본 오디오와 처리된 오디오를 비교합니다.

        Args:
            original_file (str): 원본 WAV 파일 경로
            processed_file (str): 처리된 WAV 파일 경로
            output_file (str): 비교 결과 WAV 파일 경로

        Returns:
            None: 결과는 output_file에 저장됩니다.
        """
        original = AudioSegment.from_wav(original_file)
        processed = AudioSegment.from_wav(processed_file)

        self.logger.info(f"원본 오디오 길이: {len(original)}ms, 처리된 오디오 길이: {len(processed)}ms")

        # 두 오디오의 길이를 맞춤
        max_length = max(len(original), len(processed))
        original = original + AudioSegment.silent(duration=max_length - len(original))
        processed = processed + AudioSegment.silent(duration=max_length - len(processed))

        # 원본과 처리된 오디오를 번갈아가며 1초씩 재생
        comparison = AudioSegment.empty()
        for i in range(0, max_length, 1000):
            comparison += original[i:i+1000]
            comparison += processed[i:i+1000]

        comparison.export(output_file, format="wav")
        
        self.logger.info(f"비교 오디오 길이: {len(comparison)}ms")