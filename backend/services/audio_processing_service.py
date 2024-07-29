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


from pydub import AudioSegment

class AudioProcessingService:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.soundfont_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'sf2')
        self.soundfonts = self._load_soundfonts()
        self.instrument_config = load_config('instruments')

    def _load_soundfonts(self):
        soundfonts = {}
        for file in os.listdir(self.soundfont_dir):
            if file.endswith(".sf2"):
                instrument_name = os.path.splitext(file)[0]
                soundfonts[instrument_name.lower()] = os.path.join(self.soundfont_dir, file)
        return soundfonts

    def get_instrument_program(self, instrument):
        instruments = self.instrument_config['instruments']
        default_instrument = self.instrument_config['default_instrument']
        return instruments.get(instrument.lower(), default_instrument)

    def get_instrument_program(self, instrument):
        instrument_programs = {
            'piano': 'UprightPianoKW-20220221.sf2',
            'violin': 'Valiant_Violin_V2.sf2',
            'elec_guitar_clean': 'eguitarfsbs-bridge-clean-20220911.sf2',
            'elec_guitar_dist': 'eguitarfsbs-bridge-dist1-20220911.sf2',
            'flute': 'recorder-20201205.sf2',
            'clarinet': 'clarinet-20190818.sf2',
            'harp': 'concertharp-20200702.sf2',
            'bass': 'fingerbassyr-20200813.sf2',
            'steel_string_guitar': 'fss-steelstringguitar-20200521.sf2',
            'ocarina': 'ocarina-20200726.sf2',
            'spanish_guitar': 'spanishclassicalguitar-20190618.sf2',
            'tenor_saxophone': 'tenorsaxophone-20200717.sf2',
            'timpani': 'timpani-20201121.sf2'
        }
        return instrument_programs.get(instrument.lower(), 'UprightPianoKW-20220221.sf2')

    def midi_to_wav(self, midi_path, instrument='piano'):
        """
        MIDI 파일을 WAV 파일로 변환합니다.

        Args:
            midi_path (str): 입력 MIDI 파일의 경로
            instrument (str): 사용할 악기 이름. 기본값은 'piano'

        Returns:
            str: 생성된 WAV 파일의 경로
        """
        wav_path = midi_path.rsplit('.', 1)[0] + '.wav'
        soundfont = self.get_instrument_program(instrument)
        soundfont_path = os.path.join(self.soundfont_dir, soundfont)
        
        fluidsynth_command = [
            'fluidsynth',
            '-ni',
            '-g', '1',  # gain 설정
            '-F', wav_path,
            soundfont_path,
            midi_path
        ]

        try:
            self.logger.info(f"FluidSynth 실행 명령어: {' '.join(fluidsynth_command)}")
            result = subprocess.run(fluidsynth_command, check=True, capture_output=True, text=True)
            self.logger.info(f"FluidSynth 출력: {result.stdout}")
            
            # 생성된 WAV 파일 길이 확인
            if os.path.exists(wav_path):
                audio = AudioSegment.from_wav(wav_path)
                self.logger.info(f"생성된 WAV 파일 길이: {len(audio)}ms")
            else:
                self.logger.error(f"WAV 파일이 생성되지 않았습니다: {wav_path}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FluidSynth 실행 중 오류 발생: {e}")
            self.logger.error(f"FluidSynth 오류 출력: {e.stderr}")
            raise

        return wav_path

    def amplify_wav(self, wav_path, gain_db=6):
        audio = AudioSegment.from_wav(wav_path)
        amplified_audio = audio + gain_db
        amplified_audio.export(wav_path, format="wav")
    
    def midi_to_wav(self, midi_path, instrument, gain=2.0):
        soundfont = self.get_instrument_program(instrument)
        soundfont_path = os.path.join(self.soundfont_dir, soundfont)
        
        if not os.path.exists(soundfont_path):
            self.logger.error(f"SoundFont 파일을 찾을 수 없습니다: {soundfont_path}")
            raise FileNotFoundError(f"SoundFont 파일을 찾을 수 없습니다: {soundfont_path}")
        
        wav_path = midi_path.rsplit('.', 1)[0] + '.wav'
        
        fluidsynth_command = [
            'fluidsynth',
            '-ni',
            '-g', str(gain),
            '-F', wav_path,
            soundfont_path,
            midi_path
        ]

        try:
            self.logger.info(f"FluidSynth 실행 명령어: {' '.join(fluidsynth_command)}")
            result = subprocess.run(fluidsynth_command, check=True, capture_output=True, text=True)
            self.logger.info(f"FluidSynth 출력: {result.stdout}")
            
            if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                self.logger.error(f"WAV 파일이 생성되지 않았거나 비어 있습니다: {wav_path}")
                raise RuntimeError(f"WAV 파일 생성 실패: {wav_path}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FluidSynth 실행 중 오류 발생: {e}")
            self.logger.error(f"FluidSynth 오류 출력: {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"WAV 파일 생성 중 예상치 못한 오류 발생: {str(e)}")
            raise

        return wav_path
    

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

    