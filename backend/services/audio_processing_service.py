import os
import logging
import subprocess
import tempfile
import wave
from scipy.io import wavfile
from backend.services.utils.config_loader import load_config
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, medfilt
from pydub import AudioSegment
from pydub.effects import normalize
import pywt
import yaml

from pydub import AudioSegment

class AudioProcessingService:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.soundfont_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'sf2')
        
        # 하나의 사운드 폰트에 여러가지 악기 사운드가 있음
        self.soundfont_path = os.path.join(self.soundfont_dir, 'CrisisGeneralMidi301.sf2')
        
        # 개별 악기 사운드 폰트 사용할때
        # self.soundfonts = self._load_soundfonts()

        self.instrument_config = self.load_instrument_config()

        self.gm_presets = self.load_gm_presets()

    def load_instrument_config(self):
        with open('backend/config/instruments.yaml', 'r') as file:
            return yaml.safe_load(file)

    def get_instrument_info(self, instrument_name):
        instruments = self.instrument_config['instruments']
        default = self.instrument_config['default_instrument']
        return instruments.get(instrument_name, instruments[default])

    def _load_soundfonts(self):
        soundfonts = {}
        for file in os.listdir(self.soundfont_dir):
            if file.endswith(".sf2"):
                instrument_name = os.path.splitext(file)[0]
                soundfonts[instrument_name.lower()] = os.path.join(self.soundfont_dir, file)
        return soundfonts

    def load_gm_presets(self):
        with open('backend/config/gm_presets.yaml', 'r') as file:
            return yaml.safe_load(file)

    def get_instrument_info(self, instrument_name):
        preset = self.gm_presets.get(instrument_name, self.gm_presets['Grand piano'])
        return {
            'program': preset['program'],
            'bank': preset['bank'],
            'soundfont': 'CrisisGeneralMidi301.sf2'  # 기본 사운드폰트
        }

    def get_instrument_info(self, instrument_name):
        preset = self.gm_presets.get(instrument_name, self.gm_presets['Grand piano'])
        return {
            'program': preset['program'],
            'bank': preset['bank'],
            'soundfont': 'CrisisGeneralMidi301.sf2'  # 기본 사운드폰트
        }

    def midi_to_wav(self, midi_path, output_path, instruments):
        """
        MIDI 파일을 WAV 파일로 변환합니다.

        Args:
            midi_path (str): 입력 MIDI 파일의 경로
            output_path (str): 출력 WAV 파일의 경로
            instruments (dict): 각 트랙에 사용된 악기 정보

        Returns:
            str: 생성된 WAV 파일의 경로
        """
        fluidsynth_command = [
            'fluidsynth',
            '-ni',
            '-g', '1',  # gain 설정
            '-F', output_path,
            '-O', 's16',  # 16-bit output
            self.soundfont_path,
            midi_path
        ]

        # 악기 설정 추가
        for track, instrument in instruments.items():
            instrument_info = self.get_instrument_info(instrument)
            fluidsynth_command.extend([
                f'--gain', '1',
                f'--channel-type', 'melody' if track != 'drums' else 'drums',
                f'--bank', str(instrument_info['bank']),
                f'--program', str(instrument_info['program']),
            ])

        try:
            self.logger.info(f"FluidSynth 실행 명령어: {' '.join(fluidsynth_command)}")
            result = subprocess.run(fluidsynth_command, check=True, capture_output=True, text=True)
            self.logger.info(f"FluidSynth 출력: {result.stdout}")
            
            # 생성된 WAV 파일 길이 확인
            if os.path.exists(output_path):
                audio = AudioSegment.from_wav(output_path)
                self.logger.info(f"생성된 WAV 파일 길이: {len(audio)}ms")
            else:
                self.logger.error(f"WAV 파일이 생성되지 않았습니다: {output_path}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FluidSynth 실행 중 오류 발생: {e}")
            self.logger.error(f"FluidSynth 오류 출력: {e.stderr}")
            raise

        return output_path

    def amplify_wav(self, wav_path, gain_db=6):
        audio = AudioSegment.from_wav(wav_path)
        amplified_audio = audio + gain_db
        amplified_audio.export(wav_path, format="wav")
    
    def midi_to_wav(self, midi_path, output_path, instruments):
        """
        MIDI 파일을 WAV로 변환하며, 여러 악기의 사운드폰트를 사용합니다.

        Args:
            midi_path (str): 입력 MIDI 파일 경로
            output_path (str): 출력 WAV 파일 경로
            instruments (dict): 각 파트별 선택된 악기

        Returns:
            str: 생성된 WAV 파일의 경로
        """
        soundfont_paths = set()  # 중복 제거를 위해 set 사용
        for instrument in instruments.values():
            soundfont = self.get_instrument_info(instrument)['soundfont']
            soundfont_paths.add(os.path.join(self.soundfont_dir, soundfont))
        
        soundfont_paths = list(soundfont_paths)  # set을 list로 변환
        
        fluidsynth_command = [
            'fluidsynth',
            '-ni',
            '-g', '1',  # gain 설정
            '-F', output_path,
            '-O', 's16',  # 16-bit output
            *soundfont_paths,
            midi_path
        ]

        try:
            self.logger.info(f"FluidSynth 실행 명령어: {' '.join(fluidsynth_command)}")
            result = subprocess.run(fluidsynth_command, check=True, capture_output=True, text=True)
            self.logger.info(f"FluidSynth 출력: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FluidSynth 실행 중 오류 발생: {e}")
            self.logger.error(f"FluidSynth 오류 출력: {e.stderr}")
            raise

        return output_path

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def apply_median_filter(self, data, kernel_size=3):
        return medfilt(data, kernel_size)

    
    def clean_audio(self, input_file, output_file):
        """
        입력 WAV 파일에 오디오 처리 기술을 적용하여 노이즈를 제거하고 음질을 개선합니다.

        Args:
            input_file (str): 입력 WAV 파일의 경로
            output_file (str): 출력 WAV 파일의 경로

        Returns:
            None: 결과는 output_file에 저장됩니다.
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


    def process_and_clean_audio(self, input_file, output_file):
        """
        MIDI 파일을 WAV로 변환한 후 오디오 클리닝 과정을 적용합니다.

        이 함수는 다음 단계를 수행합니다:
        1. MIDI 파일을 WAV 파일로 변환
        2. 변환된 WAV 파일에 오디오 클리닝 적용

        Args:
            input_file (str): 입력 MIDI 파일의 경로
            output_file (str): 출력 WAV 파일의 경로

        Returns:
            None: 결과는 output_file에 저장됩니다.

        주의:
            - 이 함수는 MIDI 파일을 WAV로 변환할 때 'piano' 악기를 사용합니다.
              다른 악기를 사용하려면 함수를 수정해야 합니다.
            - 오디오 클리닝 과정에서 원본 오디오의 특성이 변할 수 있습니다.
        """
        # MIDI to WAV 변환
        temp_wav_path = input_file.rsplit('.', 1)[0] + '_temp.wav'
        wav_path = self.midi_to_wav(input_file, 'piano')  # 'piano'는 예시 instrument
        
        # 오디오 클리닝
        self.clean_audio(wav_path, output_file)
        
        # 임시 WAV 파일 삭제
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        
        self.logger.info(f"Audio processed and cleaned: {output_file}")


    