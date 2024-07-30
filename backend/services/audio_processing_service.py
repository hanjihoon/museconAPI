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

    def get_instrument_program(self, instrument):
        instruments = self.instrument_config['instruments']
        default_instrument = self.instrument_config['default_instrument']
        return instruments.get(instrument.lower(), default_instrument)

    def load_gm_presets(self):
        return {
            'Grand piano': {'program': 0, 'bank': 0},
            'Bright grand piano': {'program': 1, 'bank': 0},
            'Electric grand piano': {'program': 2, 'bank': 0},
            'Honky tonk': {'program': 3, 'bank': 0},
            'Electric piano 01': {'program': 4, 'bank': 0},
            'Electric piano 02': {'program': 5, 'bank': 0},
            'Harpsichord': {'program': 6, 'bank': 0},
            'Clavinet': {'program': 7, 'bank': 0},
            'Celesta': {'program': 8, 'bank': 0},
            'Glockenspiel': {'program': 9, 'bank': 0},
            'Music box': {'program': 10, 'bank': 0},
            'Vibraphone': {'program': 11, 'bank': 0},
            'Marimba': {'program': 12, 'bank': 0},
            'Xylophone': {'program': 13, 'bank': 0},
            'Tubular bells': {'program': 14, 'bank': 0},
            'Dulcimer': {'program': 15, 'bank': 0},
            'Hammond organ': {'program': 16, 'bank': 0},
            'Percussive organ': {'program': 17, 'bank': 0},
            'Rock organ': {'program': 18, 'bank': 0},
            'Church organ': {'program': 19, 'bank': 0},
            'Reed organ': {'program': 20, 'bank': 0},
            'Accordion': {'program': 21, 'bank': 0},
            'Harmonica': {'program': 22, 'bank': 0},
            'Tango accordion': {'program': 23, 'bank': 0},
            'Nylon guitar': {'program': 24, 'bank': 0},
            'Steel guitar': {'program': 25, 'bank': 0},
            'Jazz guitar': {'program': 26, 'bank': 0},
            'Clean guitar': {'program': 27, 'bank': 0},
            'Muted guitar': {'program': 28, 'bank': 0},
            'Overdriven guitar': {'program': 29, 'bank': 0},
            'Distorted guitar': {'program': 30, 'bank': 0},
            'Guitar harmonics': {'program': 31, 'bank': 0},
            'Acoustic bass': {'program': 32, 'bank': 0},
            'Finger bass': {'program': 33, 'bank': 0},
            'Pick bass': {'program': 34, 'bank': 0},
            'Fretless bass': {'program': 35, 'bank': 0},
            'Slap bass 01': {'program': 36, 'bank': 0},
            'Slap bass 02': {'program': 37, 'bank': 0},
            'Synth bass 01': {'program': 38, 'bank': 0},
            'Synth bass 02': {'program': 39, 'bank': 0},
            'Violin': {'program': 40, 'bank': 0},
            'Viola': {'program': 41, 'bank': 0},
            'Cello': {'program': 42, 'bank': 0},
            'Contrabass': {'program': 43, 'bank': 0},
            'Tremolo strings': {'program': 44, 'bank': 0},
            'Pizzicato strings': {'program': 45, 'bank': 0},
            'Orchestral harp': {'program': 46, 'bank': 0},
            'Timpani': {'program': 47, 'bank': 0},
            'String ensemble 01': {'program': 48, 'bank': 0},
            'String ensemble 02': {'program': 49, 'bank': 0},
            'Synth strings 01': {'program': 50, 'bank': 0},
            'Synth strings 02': {'program': 51, 'bank': 0},
            'Choir aahs': {'program': 52, 'bank': 0},
            'Voice oohs': {'program': 53, 'bank': 0},
            'Synth voice': {'program': 54, 'bank': 0},
            'Orchestra hit': {'program': 55, 'bank': 0},
            'Trumpet': {'program': 56, 'bank': 0},
            'Trombone': {'program': 57, 'bank': 0},
            'Tuba': {'program': 58, 'bank': 0},
            'Muted trumpet': {'program': 59, 'bank': 0},
            'French horn': {'program': 60, 'bank': 0},
            'Brass section': {'program': 61, 'bank': 0},
            'Synth brass 01': {'program': 62, 'bank': 0},
            'Synth brass 02': {'program': 63, 'bank': 0},
            'Soprano saxophone': {'program': 64, 'bank': 0},
            'Alto saxophone': {'program': 65, 'bank': 0},
            'Tenor saxophone': {'program': 66, 'bank': 0},
            'Baritone saxophone': {'program': 67, 'bank': 0},
            'Oboe': {'program': 68, 'bank': 0},
            'English horn': {'program': 69, 'bank': 0},
            'Bassoon': {'program': 70, 'bank': 0},
            'Clarinet': {'program': 71, 'bank': 0},
            'Piccolo': {'program': 72, 'bank': 0},
            'Flute': {'program': 73, 'bank': 0},
            'Recorder': {'program': 74, 'bank': 0},
            'Pan flute': {'program': 75, 'bank': 0},
            'Blown bottle': {'program': 76, 'bank': 0},
            'Shakuhachi': {'program': 77, 'bank': 0},
            'Whistle': {'program': 78, 'bank': 0},
            'Ocarina': {'program': 79, 'bank': 0},
            'Standard drum kit': {'program': 0, 'bank': 128},
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
                f'--channel-type', 'melody',
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
    
    def midi_to_wav_multi_instrument(self, midi_path, output_path, instruments):
        """
        MIDI 파일을 WAV로 변환하며, 여러 악기의 사운드폰트를 사용합니다.

        Args:
            midi_path (str): 입력 MIDI 파일 경로
            output_path (str): 출력 WAV 파일 경로
            instruments (dict): 각 파트별 선택된 악기

        Returns:
            str: 생성된 WAV 파일의 경로
        """
        soundfont_paths = [
            os.path.join(self.soundfont_dir, self.get_instrument_program(instruments.get('melody', 'piano'))),
            os.path.join(self.soundfont_dir, self.get_instrument_program(instruments.get('bass', 'acoustic_bass'))),
            os.path.join(self.soundfont_dir, 'muldjordkit-20201018.sf2')  # 드럼은 항상 muldjordkit 사용
        ]
        
        soundfont_paths_str = ','.join(soundfont_paths)
        
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


    