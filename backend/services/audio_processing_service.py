import os
import logging
import subprocess
import tempfile
from scipy.io import wavfile
from utils.config_loader import load_config

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
            'elec_guitar_clean': 'eguitarfsbs-bridge-clean-20220911.sf2',
            'elec_guitar_dist': 'eguitarfsbs-bridge-dist1-20220911.sf2',
            'flute': 'recorder-20201205.sf2',
            'clarinet': 'clarinet-20190818.sf2',
            'harp': 'concertharp-20200702.sf2',
            'bass': 'fingerbassyr-20200813.sf2',
            'steel_string_guitar': 'fss-steelstringguitar-20200521.sf2',
            'multiinstrument': 'muldjordkit-20201018.sf2',
            'ocarina': 'ocarina-20200726.sf2',
            'spanish_guitar': 'spanishclassicalguitar-20190618.sf2',
            'tenor_saxophone': 'tenorsaxophone-20200717.sf2',
            'timpani': 'timpani-20201121.sf2'
        }
        return instrument_programs.get(instrument.lower(), 'UprightPianoKW-20220221.sf2')

    def midi_to_audio(self, midi_path, instrument):
        soundfont = self.get_instrument_program(instrument)
        soundfont_path = os.path.join(self.soundfont_dir, soundfont)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_wav = os.path.join(temp_dir, 'output.wav')
            temp_midi = os.path.join(temp_dir, 'input.mid')
            
            # # MIDI 파일을 임시 디렉토리로 복사
            # try:
            #     shutil.copy2(midi_path, temp_midi)
            #     self.logger.info(f"MIDI 파일을 임시 디렉토리로 복사: {temp_midi}")
            # except Exception as e:
            #     self.logger.error(f"MIDI 파일 복사 중 오류 발생: {str(e)}")
            #     raise

            # MIDI 파일을 임시 디렉토리로 복사
            with open(midi_path, 'rb') as f_src, open(temp_midi, 'wb') as f_dst:
                f_dst.write(f_src.read())

            fluidsynth_command = [
                'fluidsynth',
                '-ni',
                # '-r', '96000',
                # '-b', '1024',
                # '-c', '2',
                # '-g', '1',
                '-F', 
                temp_wav,
                soundfont_path,
                temp_midi
            ]

            # fluidsynth_command = [
            #     'fluidsynth',
            #     '-ni',
            #     '-F', temp_wav,
            #     soundfont_path,
            #     temp_midi
            # ]

            try:
                result = subprocess.run(fluidsynth_command, check=True, capture_output=True, text=True)
                self.logger.info(f"FluidSynth 출력: {result.stdout}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"FluidSynth 실행 중 오류 발생: {e}")
                self.logger.error(f"FluidSynth 오류 출력: {e.stderr}")
                raise

            # WAV 파일 읽기
            sample_rate, audio_data = wavfile.read(temp_wav)

        return audio_data
    
    def midi_to_wav(self, midi_path, instrument):
        soundfont = self.get_instrument_program(instrument)
        soundfont_path = os.path.join(self.soundfont_dir, soundfont)
        
        wav_path = midi_path.rsplit('.', 1)[0] + '.wav'
        
        fluidsynth_command = [
            'fluidsynth',
            '-ni',
            '-F', wav_path,
            soundfont_path,
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

        return wav_path