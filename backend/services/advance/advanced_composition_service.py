import os
import logging
import tempfile
import wave
import note_seq
import numpy as np
from note_seq.protobuf import music_pb2
from magenta.models.music_vae import configs
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import trained_model
from magenta.models.performance_rnn import performance_sequence_generator
from backend.services.audio_processing_service import AudioProcessingService
from pydub import AudioSegment
from note_seq import sequences_lib
import copy
from dotenv import load_dotenv
import yaml
import music21
from backend.services.advance.music_generator import MusicGenerator


class AdvancedCompositionService:

    GENRES = [
        "classical", "jazz", "rock", "pop", "electronic",
        "hip_hop", "r_and_b", "country", "folk", "latin"
    ]
    
    FEELS = [
        "happy", "sad", "energetic", "calm", "mysterious",
        "romantic", "nostalgic", "epic", "playful", "dark"
    ]

    def __init__(self):
        self.models = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.initialize_models()
        self.audio_service = AudioProcessingService()

        load_dotenv()  # .env 파일에서 환경 변수 로드
        self.MIN_BPM = int(os.getenv('MIN_BPM', 40))
        self.MAX_BPM = int(os.getenv('MAX_BPM', 220))
        self.DEFAULT_BPM = int(os.getenv('DEFAULT_BPM', 120))

        self.instrument_config = self.load_instrument_config()


    def initialize_models(self):
        model_configs = {
            # 'short_melody': ('cat-mel_2bar_big.tar', configs.CONFIG_MAP['cat-mel_2bar_big']),
            'long_melody': ('hierdec-mel_16bar.ckpt', configs.CONFIG_MAP['hierdec-mel_16bar']),
            # 'trio': ('hierdec-trio_16bar.tar', configs.CONFIG_MAP['hierdec-trio_16bar']),
            # 'drums': ('nade-drums_2bar_full.tar', configs.CONFIG_MAP['nade-drums_2bar_full']),
            # 'groove': ('groovae_2bar_humanize.tar', configs.CONFIG_MAP['groovae_2bar_humanize'])
        }

        for model_name, (file_name, config) in model_configs.items():
            try:
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'assets', 'models', 'cat-mel_2bar_big.ckpt')
                
                config = self.configure_data_converter(config)

                self.models[model_name] = TrainedModel(
                    config,
                    batch_size=8,
                    checkpoint_dir_or_path=model_path
                )
                self.logger.info(f"{model_name} 모델 초기화 완료")
            except Exception as e:
                self.logger.error(f"{model_name} 모델 초기화 중 오류 발생: {str(e)}")

    @staticmethod
    def configure_data_converter(config):
        if not hasattr(config, 'data_converter'):
            raise AttributeError("Config object doesn't have 'data_converter' attribute")

        dc = config.data_converter

        # steps_per_quarter 설정
        if not hasattr(dc, 'steps_per_quarter') or dc.steps_per_quarter is None:
            dc.steps_per_quarter = 480  # MIDI 표준값으로 직접 설정
        else:
            dc.steps_per_quarter = max(dc.steps_per_quarter, 480)  # 기존 값과 480 중 큰 값 사용

        # steps_per_bar 설정
        if not hasattr(dc, 'steps_per_bar') or dc.steps_per_bar is None:
            dc.steps_per_bar = dc.steps_per_quarter * 4  # 4/4 박자 기준

        return config

    def get_model_settings(self, model_name):
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"모델 '{model_name}'을(를) 찾을 수 없습니다.")
        
        config = model._config
        data_converter = config.data_converter
        
        settings = {}
        for attr in ['steps_per_quarter', 'steps_per_bar', 'slice_bars', 'max_tensors_per_input', 'max_num_steps', 'input_depth', 'input_dtype']:
            if hasattr(data_converter, attr):
                settings[attr] = getattr(data_converter, attr)
            else:
                settings[attr] = None
        
        self.logger.info(f"Model '{model_name}' settings: {settings}")
        return settings
    
    def load_instrument_config(self):
        with open('backend/config/instruments.yaml', 'r') as file:
            return yaml.safe_load(file)

    def get_instrument_info(self, instrument_name):
        instruments = self.instrument_config['instruments']
        default = self.instrument_config['default_instrument']
        return instruments.get(instrument_name, instruments[default])
    
    def apply_genre_and_feel(self, stream, genre, feel):
        genre_method = getattr(self, f"apply_{genre}_style", None)
        if genre_method:
            genre_method(stream)
        
        feel_method = getattr(self, f"apply_{feel}_feel", None)
        if feel_method:
            feel_method(stream)
        
        return stream

    def notesequence_to_stream(self, sequence):
        """
        NoteSequence를 music21 Stream으로 변환합니다.
        """
        stream = music21.stream.Stream()
        for note in sequence.notes:
            n = music21.note.Note(note.pitch)
            n.quarterLength = (note.end_time - note.start_time) / 0.5  # 가정: 4분음표 = 0.5초
            n.volume.velocity = note.velocity
            stream.insert(note.start_time, n)
        return stream

    def stream_to_notesequence(self, stream):
        """
        music21 Stream을 NoteSequence로 변환합니다.
        """
        sequence = music_pb2.NoteSequence()
        for element in stream.recurse().notesAndRests:
            if isinstance(element, music21.note.Note):
                note = sequence.notes.add()
                note.pitch = element.pitch.midi
                note.start_time = element.offset * 0.5
                note.end_time = note.start_time + (element.duration.quarterLength * 0.5)
                note.velocity = int(element.volume.velocity)
        return sequence

    def repeat_sequence(self, sequence, repeat_count):
        """
        주어진 NoteSequence를 지정된 횟수만큼 반복합니다.

        Args:
            sequence (music_pb2.NoteSequence): 반복할 시퀀스
            repeat_count (int): 반복 횟수

        Returns:
            music_pb2.NoteSequence: 반복된 시퀀스
        """
        repeated_sequence = music_pb2.NoteSequence()
        repeated_sequence.CopyFrom(sequence)
        original_total_time = sequence.total_time

        for i in range(1, repeat_count):
            for note in sequence.notes:
                new_note = repeated_sequence.notes.add()
                new_note.CopyFrom(note)
                new_note.start_time += i * original_total_time
                new_note.end_time += i * original_total_time

        repeated_sequence.total_time = original_total_time * repeat_count

        return repeated_sequence
    
    def adjust_ticks_per_quarter(self, sequence, model_name):
        """
        시퀀스의 ticks_per_quarter 값을 모델에 맞게 조정합니다.

        Args:
            sequence (music_pb2.NoteSequence): 조정할 시퀀스
            model_name (str): 모델 이름

        Returns:
            music_pb2.NoteSequence: ticks_per_quarter가 조정된 시퀀스
        """
        settings = self.get_model_settings(model_name)
        expected_ticks_per_quarter = settings.get('expected_ticks_per_quarter', 220)
        
        if sequence.ticks_per_quarter != expected_ticks_per_quarter:
            ratio = expected_ticks_per_quarter / sequence.ticks_per_quarter
            for note in sequence.notes:
                note.start_time *= ratio
                note.end_time *= ratio
            sequence.ticks_per_quarter = expected_ticks_per_quarter
        
        return sequence
    
    def adjust_sequence_length(self, sequence, target_length):
        """
        NoteSequence의 길이를 목표 길이에 맞게 조정합니다.
        """
        if sequence.total_time > target_length:
        # 시퀀스가 너무 길면 자릅니다
            sequence = note_seq.sequences_lib.extract_subsequence(
            sequence, 0, target_length)
        elif sequence.total_time < target_length:
            # 시퀀스가 너무 짧으면 반복합니다
            repeat_times = int(target_length / sequence.total_time) + 1
            sequence = self.repeat_sequence(sequence, repeat_times)
            sequence = note_seq.sequences_lib.extract_subsequence(
                sequence, 0, target_length)
        
        return sequence

    def generate_composition(self, composition_type, num_outputs=3, temperature=0.5, length=256, bpm=None, min_pitch=60, max_pitch=84, instrument=None, genre=None, feel=None):
        """
        지정된 유형의 음악 구성 요소를 생성합니다.

        Args:
            composition_type (str): 생성할 구성 요소의 유형 (예: 'melody', 'chord', 'drum', 'groove')
            num_outputs (int): 생성할 출력의 수. 기본값은 3.
            temperature (float): 생성의 무작위성을 제어하는 온도 매개변수. 기본값은 0.5.
            length (int): 생성할 시퀀스의 길이(스텝 수). 기본값은 256.
            bpm (int): 생성된 시퀀스의 템포(분당 박자 수). 지정되지 않으면 기본 BPM이 사용됩니다.
            min_pitch (int): 생성할 음표의 최소 피치. 기본값은 60 (C4).
            max_pitch (int): 생성할 음표의 최대 피치. 기본값은 84 (C6).
            instrument (str): 사용할 악기. 기본값은 None.

        Returns:
            list: 생성된 유효한 NoteSequence 객체들의 리스트
        """
        if bpm is None:
            bpm = self.DEFAULT_BPM
        bpm = self.validate_bpm(bpm)


        # 장르, 느낌 유효성 검사
        # if genre not in MusicGenerator.GENRES:
        #     raise ValueError(f"Invalid genre. Choose from: {', '.join(MusicGenerator.GENRES)}")
        # if feel not in MusicGenerator.FEELS:
        #     raise ValueError(f"Invalid feel. Choose from: {', '.join(MusicGenerator.FEELS)}")

        try:
            model = self.models.get(composition_type)
            if model is None:
                self.logger.error(f"모델 '{composition_type}'이(가) 초기화되지 않았습니다.")
                self.initialize_models()  # 모델 재초기화 시도
                model = self.models.get(composition_type)
                if model is None:
                    raise ValueError(f"모델 '{composition_type}'을(를) 초기화할 수 없습니다.")

            settings = self.get_model_settings(composition_type)
            steps_per_quarter = settings['steps_per_quarter']
            steps_per_bar = settings['steps_per_bar']

            valid_sequences = []
            max_attempts = 10  # 최대 시도 횟수

            for _ in range(max_attempts):
                
                if composition_type in ['long_melody', 'short_melody']:
                    melody_length = length
                elif composition_type in ['drums', 'groove']:
                    melody_length = length // 2  # 멜로디의 절반 길이로 설정

                sequences = model.sample(n=num_outputs, length=melody_length, temperature=temperature)

                if composition_type == 'groove':
                    processed_sequences = []
                    for i, sequence in enumerate(sequences):
                        try:
                            processed_sequence = self.process_groove_output(sequence, bpm or self.DEFAULT_BPM)
                            processed_sequences.append(processed_sequence)
                        except Exception as e:
                            self.logger.error(f"Groove 처리 중 오류 발생 (시퀀스 {i+1}): {str(e)}")
                    sequences = processed_sequences

                if not sequences:
                    self.logger.warning(f"시도 {_ + 1}: 생성된 시퀀스가 없습니다. 재시도 중...")
                    continue

                for sequence in sequences:
                    if sequence.notes:
                        # MIDI 데이터 검증 및 수정
                        sequence = self.validate_midi_data(sequence)
                        self.log_midi_details(sequence)

                        # 음표 조정
                        for note in sequence.notes:
                            # 볼륨 증가
                            note.velocity = min(note.velocity * 2, 127)
                            
                            # 시퀀스 유형 및 악기 설정
                            if composition_type == 'drums':
                                note.is_drum = True
                                note.instrument = 9  # MIDI 채널 10 (0-based index이므로 9)
                                instrument_info = self.audio_service.get_instrument_info('Standard drum kit')
                            else:
                                note.is_drum = False
                                instrument_info = self.audio_service.get_instrument_info(instrument or composition_type)
                                
                                # MIDI 채널 및 프로그램 설정
                                # if composition_type in ('long_melody','short_melody'):
                                #     note.instrument = 0  # MIDI 채널 1
                                # elif composition_type == 'bass':
                                #     note.instrument = 1  # MIDI 채널 2
                                #     note.pitch = max(28, min(note.pitch, 60))  # 베이스 음역 제한 (E1 ~ C3)
                                # else:
                                #     note.instrument = 2  # MIDI 채널 3 (필요에 따라 조정)
                                
                                note.program = instrument_info['program']
                                
                                # 피치 조정 (드럼 제외)
                                note.pitch = max(min_pitch, min(note.pitch, max_pitch))

                         # BPM 설정
                        if not sequence.tempos:
                            tempo = sequence.tempos.add()
                            tempo.qpm = bpm if bpm is not None else self.DEFAULT_BPM
                            tempo.time = 0
                        else:
                            sequence.tempos[0].qpm = bpm if bpm is not None else self.DEFAULT_BPM

                        # ticks_per_quarter 설정
                        sequence.ticks_per_quarter = 480  # 표준값으로 설정
                        # 노트의 시작 시간과 종료 시간 조정
                        ratio = 480 / 220
                        for note in sequence.notes:
                            note.start_time *= ratio
                            note.end_time *= ratio
                        sequence.total_time *= ratio

                        # 프로그램 변경 및 뱅크 선택 메시지 추가
                        # if composition_type != 'drums':
                        #     control_change = sequence.control_changes.add()
                        #     control_change.time = 0
                        #     control_change.control_number = 0  # Bank select
                        #     control_change.control_value = instrument_info['bank']

                        #     program_change = sequence.control_changes.add()
                        #     program_change.time = 0
                        #     program_change.control_number = 255  # Program change
                        #     program_change.control_value = instrument_info['program']

                        if self.validate_sequence(sequence):
                            self.log_notesequence(sequence, f"Generated {composition_type} {len(valid_sequences)+1}")
                            valid_sequences.append(sequence)

                        # 총 시간 조정
                        if sequence.total_time <= 0:
                            sequence.total_time = max(note.end_time for note in sequence.notes)

                        
                        # 장르와 느낌 적용
                        # try:
                        #     self.logger.info(f"{composition_type}에 '{genre}' 장르와 '{feel}' 느낌 적용 시작")
                        #     original_note_count = len(sequence.notes)
                        #     sequence = MusicGenerator.apply_genre_and_feel(sequence, genre, feel)
                        #     new_note_count = len(sequence.notes)
                        #     self.logger.info(f"장르와 느낌 적용 완료. 원본 음표 수: {original_note_count}, 새 음표 수: {new_note_count}")
                        #     if original_note_count == new_note_count:
                        #         self.logger.warning("장르와 느낌 적용 후 음표 수가 변경되지 않음")
                        # except Exception as e:
                        #     self.logger.error(f"장르와 느낌 적용 중 오류 발생: {str(e)}")
                        #     self.logger.exception("상세 오류 정보:")

                        self.log_notesequence(sequence, f"Generated {composition_type} {len(valid_sequences)+1}")
                        self.logger.info(f"생성된 {composition_type} {len(valid_sequences)+1} 정보: 음표 수: {len(sequence.notes)}, 총 시간: {sequence.total_time}, BPM: {bpm}")
                        
                        if self.validate_sequence(sequence):
                            valid_sequences.append(sequence)
                        else:
                            self.logger.warning(f"유효하지 않은 시퀀스가 생성되었습니다: {sequence}")
                    else:
                        self.logger.warning(f"빈 {composition_type} 시퀀스가 생성되었습니다.")

                if len(valid_sequences) >= num_outputs:
                    break
            
            if not valid_sequences:
                raise ValueError(f"유효한 {composition_type} 시퀀스가 생성되지 않았습니다.")

            self.logger.info(f"{composition_type} 생성 완료 - 유효한 시퀀스 수: {len(valid_sequences)}")
            return valid_sequences[:num_outputs]

        except Exception as e:
            self.logger.error(f"{composition_type} 생성 중 오류 발생: {str(e)}")
            self.logger.exception("상세 오류 정보:")
            raise

    def apply_groove(self, drum_sequence):
        """
        드럼 시퀀스에 그루브를 적용합니다.

        이 함수는 입력된 드럼 시퀀스를 작은 세그먼트로 나누고, 각 세그먼트에 그루브를 적용한 후
        다시 하나의 시퀀스로 결합합니다. 그루브 적용은 'groove' 모델을 사용하여 수행됩니다.

        Args:
            drum_sequence (music_pb2.NoteSequence): 그루브를 적용할 원본 드럼 시퀀스

        Returns:
            music_pb2.NoteSequence: 그루브가 적용된 드럼 시퀀스

        Raises:
            ValueError: 그루브 모델이 초기화되지 않았거나 사용할 수 없는 경우
            Exception: 그루브 적용 과정에서 예상치 못한 오류가 발생한 경우

        Notes:
            - 이 함수는 드럼 시퀀스를 2마디(32스텝) 단위로 분할하여 처리합니다.
            - 그루브 적용 후 각 세그먼트의 총 시간이 올바르게 설정되었는지 확인합니다.
            - 유효한 그루브 시퀀스가 하나도 생성되지 않으면 원본 시퀀스를 반환합니다.
        """
        groove_model = self.models.get('groove')
        if groove_model is None:
            self.logger.warning("그루브 모델이 초기화되지 않았습니다. 그루브를 적용할 수 없습니다.")
            return drum_sequence
        
        try:
            self.logger.info(f"원본 드럼 시퀀스 - 노트 수: {len(drum_sequence.notes)}, 총 시간: {drum_sequence.total_time}")
            
            # 시퀀스를 2마디 단위로 자르기
            sub_sequences = self.split_sequence(drum_sequence, 32)  # 32 steps = 2 bars
            self.logger.info(f"분할된 서브시퀀스 수: {len(sub_sequences)}")
            
            grooved_sequences = []
            for i, sub_seq in enumerate(sub_sequences):
                self.logger.info(f"서브시퀀스 {i+1} - 노트 수: {len(sub_seq.notes)}, 총 시간: {sub_seq.total_time}")
                
                grooved_sequence = groove_model.humanize(sub_seq)
                self.logger.info(f"그루브 적용 후 서브시퀀스 {i+1} - 노트 수: {len(grooved_sequence.notes)}, 총 시간: {grooved_sequence.total_time}")
                
                if grooved_sequence.notes:
                    end_time = max(note.end_time for note in grooved_sequence.notes)
                    new_sequence = music_pb2.NoteSequence()
                    new_sequence.CopyFrom(grooved_sequence)
                    new_sequence.total_time = end_time
                    grooved_sequences.append(new_sequence)

            if not grooved_sequences:
                self.logger.warning("그루브 적용 후 유효한 시퀀스가 없습니다.")
                return drum_sequence

            # 모든 서브시퀀스 합치기
            final_sequence = self.concatenate_sequences(grooved_sequences)
            self.logger.info(f"최종 그루브 시퀀스 - 노트 수: {len(final_sequence.notes)}, 총 시간: {final_sequence.total_time}")
            
            return final_sequence
        except Exception as e:
            self.logger.error(f"그루브 적용 중 오류 발생: {str(e)}")
            return drum_sequence

    def split_sequence(self, sequence, steps_per_segment):
        """
        NoteSequence를 지정된 스텝 수에 따라 여러 개의 세그먼트로 분할합니다.

        Args:
            sequence (music_pb2.NoteSequence): 분할할 원본 시퀀스
            steps_per_segment (int): 각 세그먼트의 스텝 수

        Returns:
            list: 분할된 NoteSequence 세그먼트들의 리스트
        """
        segments = []
        
        # 시퀀스의 총 길이 (초 단위)를 계산
        total_time = sequence.total_time
        
        # 스텝당 시간 계산 (4분 음표 기준)
        if sequence.tempos:
            quarter_note_duration = 60.0 / sequence.tempos[0].qpm
        else:
            quarter_note_duration = 0.5  # 기본값: 120 BPM
        step_duration = quarter_note_duration / 4  # 16분 음표 기준
        
        # 세그먼트당 시간 계산
        segment_duration = steps_per_segment * step_duration
        
        # 시퀀스를 세그먼트로 분할
        for start_time in np.arange(0, total_time, segment_duration):
            end_time = min(start_time + segment_duration, total_time)
            
            # 새 세그먼트 생성
            segment = music_pb2.NoteSequence()
            segment.CopyFrom(sequence)
            
            # 세그먼트 내의 노트만 유지
            segment.notes[:] = [note for note in segment.notes 
                                if start_time <= note.start_time < end_time]
            
            # 노트의 시작 시간과 종료 시간 조정
            for note in segment.notes:
                note.start_time -= start_time
                note.end_time = min(note.end_time - start_time, segment_duration)
            
            # 세그먼트의 총 시간 설정
            segment.total_time = end_time - start_time
            
            # 템포 정보 복사 (필요한 경우)
            if sequence.tempos:
                segment.tempos[0].time = 0
            
            segments.append(segment)
        
        return segments

    def concatenate_sequences(self, sequences):
        """
        여러 개의 NoteSequence를 하나의 연속된 시퀀스로 연결합니다.

        Args:
            sequences (list): NoteSequence 객체들의 리스트

        Returns:
            music_pb2.NoteSequence: 연결된 하나의 NoteSequence
        """
        if not sequences:
            return music_pb2.NoteSequence()

        combined_sequence = music_pb2.NoteSequence()
        combined_sequence.CopyFrom(sequences[0])
        
        current_end_time = sequences[0].total_time
        
        for seq in sequences[1:]:
            # 노트 추가
            for note in seq.notes:
                new_note = combined_sequence.notes.add()
                new_note.CopyFrom(note)
                new_note.start_time += current_end_time
                new_note.end_time += current_end_time
            
            # 컨트롤 변경 추가 (있는 경우)
            for control in seq.control_changes:
                new_control = combined_sequence.control_changes.add()
                new_control.CopyFrom(control)
                new_control.time += current_end_time
            
            # 피치 밴드 추가 (있는 경우)
            for pitch_bend in seq.pitch_bends:
                new_pitch_bend = combined_sequence.pitch_bends.add()
                new_pitch_bend.CopyFrom(pitch_bend)
                new_pitch_bend.time += current_end_time
            
            # 총 시간 업데이트
            current_end_time += seq.total_time
        
        # 최종 시퀀스의 총 시간 설정
        combined_sequence.total_time = current_end_time
        
        # 템포 정보 설정 (모든 시퀀스가 동일한 템포를 가진다고 가정)
        if combined_sequence.tempos and len(combined_sequence.tempos) > 1:
            del combined_sequence.tempos[1:]
        
        # 박자 표시 설정 (모든 시퀀스가 동일한 박자를 가진다고 가정)
        if combined_sequence.time_signatures and len(combined_sequence.time_signatures) > 1:
            del combined_sequence.time_signatures[1:]
        
        return combined_sequence

    def save_midi(self, sequence, filename, instruments):
        """
        NoteSequence를 MIDI 파일로 저장하고 WAV 파일로 변환합니다.

        Args:
            sequence (music_pb2.NoteSequence): 저장할 NoteSequence
            filename (str): 저장할 파일 이름 (확장자 제외)
            instruments (dict): 각 파트별 악기 선택 (예: {'melody': 'piano', 'bass': 'acoustic_bass', 'drums': 'drum_kit'})

        Returns:
            tuple: (midi_path, wav_path)
        """
        try:
            midi_path = filename + '.mid'
            wav_path = filename + '.wav'
            
            # MIDI 파일 저장 전 시퀀스 검증
            sequence = self.validate_midi_data(sequence)
            
            # MIDI 파일 저장 시도
            try:
                note_seq.sequence_proto_to_midi_file(sequence, midi_path)
            except Exception as midi_error:
                self.logger.error(f"MIDI 파일 저장 중 오류 발생: {str(midi_error)}")
                # MIDI 시퀀스의 세부 정보 로깅
                self.log_midi_details(sequence)
                raise

            self.logger.info(f"MIDI 파일 저장 완료: {midi_path}")
            
            if not os.path.exists(midi_path):
                raise FileNotFoundError(f"MIDI 파일을 찾을 수 없습니다: {midi_path}")
            
            # MIDI to WAV 변환
            wav_path = self.audio_service.midi_to_wav(midi_path, wav_path, instruments)
            
            self.logger.info(f"WAV 파일 생성 완료: {wav_path}")

            return midi_path, wav_path
        except Exception as e:
            self.logger.error(f"파일 저장 중 오류 발생: {str(e)}")
            raise


    def pad_sequence(self, sequence, desired_length):
        """
        NoteSequence를 원하는 길이로 패딩합니다.

        Args:
            sequence (note_seq.NoteSequence): 패딩할 NoteSequence
            desired_length (float): 원하는 시퀀스 길이 (초 단위)

        Returns:
            note_seq.NoteSequence: 패딩된 NoteSequence
        """
        if sequence.total_time >= desired_length:
            return sequence

        padded_sequence = copy.deepcopy(sequence)
        padded_sequence.total_time = desired_length
        return padded_sequence
    
    def log_sequence_info(self, sequence, description):
        """
        NoteSequence의 메타데이터를 로깅합니다 (노트 정보 제외).

        Args:
            sequence (music_pb2.NoteSequence): 로깅할 시퀀스
            description (str): 로그 설명
        """
        info = f"{description}:\n"
        info += f"ticks_per_quarter: {sequence.ticks_per_quarter}\n"
        
        if sequence.time_signatures:
            time_sig = sequence.time_signatures[0]
            info += f"time_signature: {time_sig.numerator}/{time_sig.denominator}\n"
        
        if sequence.tempos:
            info += f"tempo: {sequence.tempos[0].qpm} qpm\n"
        
        info += f"total_time: {sequence.total_time:.2f} seconds\n"
        info += f"total_notes: {len(sequence.notes)}\n"
        
        if hasattr(sequence, 'subsequence_info'):
            info += f"subsequence_info: end_time_offset = {sequence.subsequence_info.end_time_offset:.2f}\n"
        
        self.logger.info(info)

    def log_notesequence(self, sequence, description):
        self.logger.info(f"{description}:")
        self.logger.info(f"Total time: {sequence.total_time}")
        self.logger.info(f"Tempo: {sequence.tempos[0].qpm if sequence.tempos else 'Not specified'}")
        self.logger.info(f"Total notes: {len(sequence.notes)}")
        for i, note in enumerate(sequence.notes[:10]):
            self.logger.info(f"Note {i+1}: pitch={note.pitch}, start_time={note.start_time:.3f}, end_time={note.end_time:.3f}, velocity={note.velocity}")
        if len(sequence.notes) > 10:
            self.logger.info("...")

    def prepare_sequence_for_model(self, sequence, model_name, num_steps=256):
        """
        NoteSequence를 모델이 기대하는 형식으로 준비합니다.

        Args:
            sequence (note_seq.NoteSequence): 준비할 NoteSequence
            model_name (str): 사용할 모델 이름
            num_steps (int): 원하는 시퀀스의 길이

        Returns:
            note_seq.NoteSequence: 준비된 NoteSequence
        """
        settings = self.get_model_settings(model_name)
        steps_per_quarter = settings.get('steps_per_quarter', 4)
        max_num_steps = settings.get('max_num_steps')

        if steps_per_quarter is None:
            self.logger.warning(f"모델 '{model_name}'의 steps_per_quarter가 None입니다. 기본값 4를 사용합니다.")
            steps_per_quarter = 4

        qpm = 120.0
        seconds_per_step = 60 / (qpm * steps_per_quarter)
        
        # max_num_steps가 None인 경우를 처리
        if max_num_steps is None:
            self.logger.warning(f"모델 '{model_name}'의 max_num_steps가 None입니다. num_steps를 사용합니다.")
            max_num_steps = num_steps
        
        total_time = min(num_steps, max_num_steps) * seconds_per_step

        prepared_sequence = music_pb2.NoteSequence()
        prepared_sequence.CopyFrom(sequence)

        # BPM 설정
        while prepared_sequence.tempos:
            prepared_sequence.tempos.pop()
        prepared_sequence.tempos.add(qpm=qpm)

        # ticks_per_quarter 설정
        prepared_sequence.ticks_per_quarter = steps_per_quarter * 220 // 4

        # 시퀀스 길이 조정
        prepared_sequence = self.adjust_sequence_length(prepared_sequence, total_time)

        # 드럼 시퀀스인 경우 추가 처리
        if model_name == 'drums':
            prepared_sequence = self.prepare_drum_sequence(prepared_sequence, steps_per_quarter)
        else:
            prepared_sequence = self.prepare_melody_sequence(prepared_sequence, steps_per_quarter)

        self.log_sequence_info(prepared_sequence, f"Prepared {model_name} sequence")
        return prepared_sequence

    def prepare_drum_sequence(self, sequence, steps_per_quarter):
        # 드럼 시퀀스에 대한 특별한 처리
        for note in sequence.notes:
            note.is_drum = True
            note.instrument = 9  # MIDI 드럼 채널
        return sequence

    def prepare_melody_sequence(self, sequence, steps_per_quarter):
        # 멜로디 시퀀스에 대한 특별한 처리
        for note in sequence.notes:
            note.is_drum = False
            note.instrument = 0  # MIDI 피아노 채널
        return sequence

    # def prepare_sequence_for_model(self, sequence, model_name, num_steps=256):
    #     """
    #     NoteSequence를 모델이 기대하는 형식으로 준비합니다.

    #     Args:
    #         sequence (note_seq.NoteSequence): 준비할 NoteSequence
    #         model_name (str): 사용할 모델 이름
    #         num_steps (int): 원하는 시퀀스의 길이

    #     Returns:
    #         note_seq.NoteSequence: 준비된 NoteSequence
    #     """
    #     settings = self.get_model_settings(model_name)
    #     steps_per_quarter = settings.get('steps_per_quarter', 4)
        
    #     if steps_per_quarter is None:
    #         self.logger.warning(f"모델 '{model_name}'의 steps_per_quarter가 None입니다. 기본값 4를 사용합니다.")
    #         steps_per_quarter = 4

    #     qpm = 120.0
    #     seconds_per_step = 60 / (qpm * steps_per_quarter)
    #     total_time = num_steps * seconds_per_step

    #     prepared_sequence = music_pb2.NoteSequence()
    #     prepared_sequence.CopyFrom(sequence)
        
    #     # tempos를 초기화하고 새로운 tempo 추가
    #     del prepared_sequence.tempos[:]
    #     prepared_sequence.tempos.add(qpm=qpm)

    #     # ticks_per_quarter 조정
    #     prepared_sequence = self.adjust_ticks_per_quarter(prepared_sequence, model_name)

    #     prepared_sequence = self.validate_and_fix_notesequence(prepared_sequence, model_name)

    #     self.logger.info(f"원본 시퀀스 정보: 음표 수: {len(sequence.notes)}, 총 시간: {sequence.total_time:.2f}초")
    #     self.log_sequence_info(sequence, "원본 시퀀스 정보")
        
    #     # 시퀀스 길이 조정
    #     expected_length = num_steps * seconds_per_step
    #     if prepared_sequence.total_time < expected_length:
    #         repeat_count = int(expected_length / prepared_sequence.total_time) + 1
    #         self.logger.info(f"시퀀스가 짧아 {repeat_count}번 반복합니다.")
    #         prepared_sequence = self.repeat_sequence(prepared_sequence, repeat_count)

    #     prepared_sequence = note_seq.sequences_lib.extract_subsequence(
    #         prepared_sequence, 0, expected_length)
        
    #     self.log_sequence_info(prepared_sequence, "조정된 시퀀스 정보")
        
    #     notes_to_remove = []
    #     for note in prepared_sequence.notes:
    #         if note.start_time >= total_time:
    #             notes_to_remove.append(note)
    #         elif note.end_time > total_time:
    #             note.end_time = total_time
        
    #     for note in notes_to_remove:
    #         prepared_sequence.notes.remove(note)

    #     prepared_sequence.total_time = min(prepared_sequence.total_time, total_time)

    #     # 시간 시그니처 추가 (4/4 박자 가정)
    #     if not prepared_sequence.time_signatures:
    #         time_signature = prepared_sequence.time_signatures.add()
    #         time_signature.numerator = 4
    #         time_signature.denominator = 4
    #         time_signature.time = 0

    #     self.logger.info(f"전처리된 시퀀스 정보: 음표 수: {len(prepared_sequence.notes)}, "
    #                     f"총 시간: {prepared_sequence.total_time:.2f}초, "
    #                     f"ticks_per_quarter: {prepared_sequence.ticks_per_quarter}, "
    #                     f"템포: {prepared_sequence.tempos[0].qpm}")

    #     return prepared_sequence
    
    def interpolate(self, start, end, num_steps):
        """
        두 잠재 벡터 사이를 선형 보간합니다.

        Args:
            start (np.ndarray): 시작 잠재 벡터
            end (np.ndarray): 끝 잠재 벡터
            num_steps (int): 보간 단계 수

        Returns:
            list: 보간된 잠재 벡터들의 리스트
        """
        interpolated = []
        for step in range(num_steps):
            alpha = step / (num_steps - 1)
            interpolated_vector = (1 - alpha) * start + alpha * end
            # 벡터 형태 확인 및 조정
            if interpolated_vector.ndim == 1:
                interpolated_vector = interpolated_vector.reshape(1, -1)
            self.logger.info(f"보간된 벡터 {step+1}: 형태 {interpolated_vector.shape}, 최소값 {interpolated_vector.min():.4f}, 최대값 {interpolated_vector.max():.4f}, 평균 {interpolated_vector.mean():.4f}")
            interpolated.append(interpolated_vector)
        return interpolated
    
    def validate_notesequence(self, sequence):
        """NoteSequence가 유효한지 검증합니다."""
        if not sequence.notes:
            raise ValueError("NoteSequence에 음표가 없습니다.")
        if sequence.total_time <= 0:
            raise ValueError("NoteSequence의 총 시간이 0 이하입니다.")
        if not sequence.tempos:
            raise ValueError("NoteSequence에 템포 정보가 없습니다.")
        if sequence.ticks_per_quarter <= 0:
            raise ValueError("ticks_per_quarter가 0 이하입니다.")

    def validate_and_fix_notesequence(self, sequence, model_name):
        """
        NoteSequence의 유효성을 검사하고 필요한 경우 수정합니다.
        """
        settings = self.get_model_settings(model_name)
        steps_per_quarter = settings.get('steps_per_quarter', 4)
        
        # ticks_per_quarter 조정
        if sequence.ticks_per_quarter != steps_per_quarter * 220 // 4:
            sequence.ticks_per_quarter = steps_per_quarter * 220 // 4
        
        # 시간 서명 확인 및 추가
        if not sequence.time_signatures:
            time_signature = sequence.time_signatures.add()
            time_signature.numerator = 4
            time_signature.denominator = 4
            time_signature.time = 0
        
        # 템포 확인 및 추가
        if not sequence.tempos:
            tempo = sequence.tempos.add()
            tempo.qpm = 120.0
            tempo.time = 0
        
        # 노트가 없는 경우 더미 노트 추가
        if not sequence.notes:
            note = sequence.notes.add()
            note.pitch = 60
            note.velocity = 80
            note.start_time = 0
            note.end_time = 0.5
            note.is_drum = False
            sequence.total_time = 0.5
        
        return sequence

    def interpolate_melodies(self, start_melody, end_melody, num_steps=5, length=256, bpm=None, instruments=None):
        """
        두 멜로디 사이를 보간하여 새로운 멜로디들을 생성합니다.

        Args:
            start_melody (NoteSequence): 시작 멜로디
            end_melody (NoteSequence): 끝 멜로디
            num_steps (int): 보간 단계 수 (기본값: 5)
            length (int): 생성할 멜로디의 길이 (기본값: 256)
            bpm (int): 생성할 멜로디의 템포 (기본값: None)

        Returns:
            list: 보간된 멜로디들의 리스트 (NoteSequence 객체들)

        Raises:
            ValueError: 입력 멜로디가 유효하지 않을 경우
        """
        if bpm is None:
            bpm = self.DEFAULT_BPM
        bpm = self.validate_bpm(bpm)

        if instruments is None:
            instruments = {
                'melody': 'piano',
                'bass': 'acoustic_bass',
                'drums': 'drum_kit'
            }

        data_converter = self.models['long_melody']._config.data_converter
        self.logger.info(f"데이터 컨버터 설정: {data_converter.__dict__}")
        self.log_sequence_info(start_melody, "시작 멜로디 정보")
        self.log_sequence_info(end_melody, "끝 멜로디 정보")
        try:

            # 드럼 시퀀스인지 확인
            is_drum = start_melody.notes[0].is_drum if start_melody.notes else False


            # 입력 멜로디 유효성 검사 및 전처리
            if not isinstance(start_melody, music_pb2.NoteSequence) or not isinstance(end_melody, music_pb2.NoteSequence):
                raise ValueError("시작 또는 끝 멜로디가 올바른 NoteSequence 형식이 아닙니다.")

            if not start_melody.notes or not end_melody.notes:
                raise ValueError("시작 또는 끝 멜로디가 비어 있습니다.")

            # 적절한 모델 선택
            model_name = 'drums' if is_drum else 'long_melody'
            self.logger.info(f"Using model: {model_name}")

            # 시퀀스 준비
            start_melody = self.prepare_sequence_for_model(start_melody, model_name, length)
            end_melody = self.prepare_sequence_for_model(end_melody, model_name, length)


            # 인코딩
            model = self.models[model_name]
            z_start = model.encode([start_melody])
            z_end = model.encode([end_melody])

            self.log_sequence_info(start_melody, "전처리된 시작 멜로디 정보")
            self.log_sequence_info(end_melody, "전처리된 끝 멜로디 정보")

            # 마디 수 계산 (4/4 박자 가정)
            quarters_per_bar = 4
            start_melody_bars = start_melody.total_time / (60 / start_melody.tempos[0].qpm * quarters_per_bar)
            end_melody_bars = end_melody.total_time / (60 / end_melody.tempos[0].qpm * quarters_per_bar)

            self.logger.info(f"Start melody bars: {start_melody_bars:.2f}")
            self.logger.info(f"End melody bars: {end_melody_bars:.2f}")

            # 마디 수 조정 (필요한 경우)
            max_steps = getattr(data_converter, 'max_num_steps', length)
            target_steps = min(length, max_steps)
            target_bars = target_steps / 16  # 16 steps per bar

            if start_melody_bars > target_bars or end_melody_bars > target_bars:
                target_seconds = target_bars * (60 / start_melody.tempos[0].qpm * quarters_per_bar)
                start_melody = note_seq.sequences_lib.extract_subsequence(start_melody, 0, target_seconds)
                end_melody = note_seq.sequences_lib.extract_subsequence(end_melody, 0, target_seconds)
                self.logger.info(f"Melodies trimmed to {target_bars} bars")

            self.logger.info(f"보간 설정: num_steps={num_steps}, length={length}")
            self.logger.info(f"준비된 시작 멜로디: 음표 수: {len(start_melody.notes)}, 총 시간: {start_melody.total_time:.2f}초")
            self.logger.info(f"준비된 끝 멜로디: 음표 수: {len(end_melody.notes)}, 총 시간: {end_melody.total_time:.2f}초")

            # 멜로디 인코딩
            try:
                z_start = self.models['long_melody'].encode([start_melody])
                self.logger.info("시작 멜로디 인코딩 성공")
            except Exception as e:
                self.logger.error(f"시작 멜로디 인코딩 중 오류 발생: {str(e)}")
                self.logger.info("문제의 NoteSequence 정보:")
                self.log_sequence_info(start_melody, "문제의 시작 멜로디")
                if hasattr(e, 'sequence_info'):
                    self.logger.info(f"오류 관련 시퀀스 정보: {e.sequence_info}")
                raise

            try:
                z_end = self.models['long_melody'].encode([end_melody])
                self.logger.info(f"끝 멜로디 인코딩 성공. 형태: {z_end[0].shape if isinstance(z_end, tuple) else z_end.shape}")
            except trained_model.NoExtractedExamplesError as e:
                error_message = str(e)
                # NoteSequence 정보에서 notes 부분 제거
                error_parts = error_message.split('notes {', 1)
                if len(error_parts) > 1:
                    error_message = error_parts[0] + '... (notes 정보 생략)'
                self.logger.error(f"멜로디 인코딩 중 오류 발생: {error_message}")
                raise

            self.logger.info(f"z_start 타입: {type(z_start)}, z_end 타입: {type(z_end)}")
            
            # z_start와 z_end가 튜플인 경우 첫 번째 요소 사용
            if isinstance(z_start, tuple):
                z_start = z_start[0]
            if isinstance(z_end, tuple):
                z_end = z_end[0]

            self.logger.info(f"z_start 형태: {z_start.shape}, z_end 형태: {z_end.shape}")

            # numpy 배열로 변환
            z_start = np.array(z_start)
            z_end = np.array(z_end)

            # 차원 확인 및 조정
            if z_start.ndim == 1:
                z_start = z_start[np.newaxis, :]
            if z_end.ndim == 1:
                z_end = z_end[np.newaxis, :]

            self.logger.info(f"조정된 z_start 형태: {z_start.shape}, z_end 형태: {z_end.shape}")
            self.logger.info(f"z_start 값 범위: {z_start.min()} ~ {z_start.max()}")
            self.logger.info(f"z_end 값 범위: {z_end.min()} ~ {z_end.max()}")

            # 멜로디 보간
            interpolated = self.interpolate(z_start, z_end, num_steps)

            # 보간된 멜로디 처리
            valid_interpolated = []
            for i, latent_vector in enumerate(interpolated):
                try:
                    self.logger.info(f"보간된 벡터 {i+1} 형태: {latent_vector.shape}")
                    self.logger.info(f"보간된 벡터 {i+1} 값 범위: {latent_vector.min()} ~ {latent_vector.max()}")

                    # 잠재 벡터를 NoteSequence로 디코딩
                    decoded = self.models['long_melody'].decode(latent_vector, length=length)
                    self.logger.info(f"디코딩된 멜로디 {i+1} 타입: {type(decoded)}")

                    if isinstance(decoded, list):
                        # 리스트인 경우 첫 번째 요소 사용
                        melody = decoded[0] if decoded else None
                    elif isinstance(decoded, music_pb2.NoteSequence):
                        melody = decoded
                    else:
                        self.logger.warning(f"예상치 못한 디코딩 결과 타입: {type(decoded)}")
                        melody = None

                    if isinstance(melody, music_pb2.NoteSequence):
                        self.logger.info(f"디코딩된 멜로디 {i+1} 정보: 음표 수: {len(melody.notes)}, 총 시간: {melody.total_time:.2f}초")
                        if len(melody.notes) > 0:
                            self.logger.info(f"BPM 조절 시작: {bpm}")
                            # BPM 조절
                            adjusted_melody = self.adjust_tempo(melody, bpm)
                            self.logger.info(f"BPM 조절 완료: {adjusted_melody.tempos[0].qpm}")
                            
                            self.logger.info("반주 생성 시작")
                            # 반주 생성
                            bass, drums = self.generate_accompaniment(adjusted_melody, bpm, instruments)
                            self.logger.info(f"반주 생성 완료: 베이스 음표 수: {len(bass.notes)}, 드럼 음표 수: {len(drums.notes)}")
                            
                            self.logger.info("시퀀스 결합 시작")
                            # 멜로디, 베이스, 드럼 결합
                            combined_sequence = self.combine_sequences(adjusted_melody, bass, drums)
                            self.logger.info(f"시퀀스 결합 완료: 총 음표 수: {len(combined_sequence.notes)}")
                            
                            self.logger.info(f"결합된 시퀀스 {i+1} 정보: 음표 수: {len(combined_sequence.notes)}, 총 시간: {combined_sequence.total_time:.2f}초, BPM: {bpm}")
                            valid_interpolated.append(combined_sequence)
                        else:
                            self.logger.warning(f"보간된 멜로디 {i+1}에 음표가 없습니다.")
                    else:
                        self.logger.warning(f"보간된 멜로디 {i+1}이(가) 유효한 NoteSequence가 아닙니다. 타입: {type(melody)}")

                except Exception as e:
                    self.logger.error(f"멜로디 {i+1} 디코딩 중 오류 발생: {str(e)}")
                    self.logger.exception("디코딩 오류 상세 정보:")

            if not valid_interpolated:
                raise ValueError("유효한 보간 멜로디가 생성되지 않았습니다.")
            
            for i, melody in enumerate(valid_interpolated):
                self.log_notesequence(melody, f"Interpolated melody {i+1}")

            return valid_interpolated

        except Exception as e:
            self.logger.error(f"멜로디 보간 중 오류 발생: {str(e)}")
            self.logger.exception("상세 오류 정보:")
            raise

    def validate_bpm(self, bpm):
        if not (self.MIN_BPM <= bpm <= self.MAX_BPM):
            raise ValueError(f"BPM은 {self.MIN_BPM}에서 {self.MAX_BPM} 사이여야 합니다.")
        return bpm
    
    def check_wav_file(self, wav_path):
        try:
            with wave.open(wav_path, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                self.logger.info(f"WAV 파일 정보 - 채널: {n_channels}, 샘플 너비: {sample_width}, "
                                f"프레임 레이트: {framerate}, 프레임 수: {n_frames}")
                
                if n_frames == 0:
                    self.logger.error(f"WAV 파일에 오디오 데이터가 없습니다: {wav_path}")
        except Exception as e:
            self.logger.error(f"WAV 파일 확인 중 오류 발생: {str(e)}")
    
    def adjust_sequence_length(self, sequence, target_time):
        """
        NoteSequence의 길이를 목표 시간에 맞게 조정합니다.

        Args:
            sequence (music_pb2.NoteSequence): 조정할 시퀀스
            target_time (float): 목표 시간 (초)

        Returns:
            music_pb2.NoteSequence: 길이가 조정된 시퀀스
        """
        if sequence.total_time == target_time:
            return sequence
        
        adjusted_sequence = music_pb2.NoteSequence()
        adjusted_sequence.CopyFrom(sequence)
        
        if sequence.total_time < target_time:
            # 시퀀스가 짧으면 늘림
            time_ratio = target_time / sequence.total_time
            for note in adjusted_sequence.notes:
                note.start_time *= time_ratio
                note.end_time *= time_ratio
        else:
            # 시퀀스가 길면 자름
            adjusted_sequence.notes[:] = [note for note in adjusted_sequence.notes if note.start_time < target_time]
            for note in adjusted_sequence.notes:
                if note.end_time > target_time:
                    note.end_time = target_time
        
        adjusted_sequence.total_time = target_time
        return adjusted_sequence

    def adjust_tempo(self, melody, target_bpm):
        """
        멜로디의 템포를 조절합니다.

        
        Parameters:
        - melody (NoteSequence): 조절할 멜로디
        - target_bpm (int): 목표 BPM

        Returns:
        - NoteSequence: 템포가 조절된 멜로디
        """
        if not melody.tempos:
            original_bpm = self.DEFAULT_BPM
        else:
            original_bpm = melody.tempos[0].qpm
        tempo_ratio = target_bpm / original_bpm

        adjusted_melody = music_pb2.NoteSequence()
        adjusted_melody.CopyFrom(melody)
        while adjusted_melody.tempos:
            adjusted_melody.tempos.pop()
        adjusted_melody.tempos.add(qpm=target_bpm)

        ticks_ratio = 480 / adjusted_melody.ticks_per_quarter
        adjusted_melody.ticks_per_quarter = 480

        for note in adjusted_melody.notes:
            note.start_time *= tempo_ratio * ticks_ratio
            note.end_time *= tempo_ratio * ticks_ratio

        adjusted_melody.total_time *= tempo_ratio * ticks_ratio

        return adjusted_melody
    

    def save_midi_with_instruments(self, sequence, filename, instruments):
        """
        NoteSequence를 MIDI 파일로 저장하고, 선택된 악기로 WAV 파일을 생성합니다.

        Args:
            sequence (music_pb2.NoteSequence): 저장할 NoteSequence
            filename (str): 저장할 파일 이름 (확장자 제외)
            instruments (dict): 각 파트별 악기 선택

        Returns:
            tuple: (midi_path, wav_path)
        """
        try:
            midi_path = filename + '.mid'
            wav_path = filename + '.wav'
            
            note_seq.sequence_proto_to_midi_file(sequence, midi_path)
            self.logger.info(f"MIDI 파일 저장 완료: {midi_path}")
            
            if not os.path.exists(midi_path):
                raise FileNotFoundError(f"MIDI 파일을 찾을 수 없습니다: {midi_path}")
            
            # MIDI to WAV 변환 (선택된 악기 사용)
            wav_path = self.audio_service.midi_to_wav_with_instruments(midi_path, wav_path, instruments)
            
            self.logger.info(f"WAV 파일 생성 완료: {wav_path}")

            return midi_path, wav_path
        except Exception as e:
            self.logger.error(f"파일 저장 중 오류 발생: {str(e)}")
            raise

    def combine_sequences(self, *sequences):
        """
        여러 개의 NoteSequence를 하나로 결합합니다.

        Args:
            *sequences: 결합할 NoteSequence 객체들

        Returns:
            music_pb2.NoteSequence: 결합된 시퀀스
        """
        self.logger.info(f"시퀀스 결합 시작: 입력 시퀀스 수: {len(sequences)}")
        combined_sequence = music_pb2.NoteSequence()
    
        for i, seq in enumerate(sequences):
            for note in seq.notes:
                new_note = combined_sequence.notes.add()
                new_note.MergeFrom(note)
                new_note.instrument = i
                new_note.program = i
                if i == 2:  # 드럼 트랙 (인덱스에 따라 조정 필요)
                    new_note.velocity = min(127, int(note.velocity * 1.5))  # 드럼 볼륨 증가
        
        combined_sequence.total_time = max(seq.total_time for seq in sequences)
        
        if sequences and sequences[0].tempos:
            combined_sequence.tempos.extend(sequences[0].tempos)
        
        return combined_sequence
    
    def validate_midi_data(self, sequence):
        """
        MIDI 시퀀스의 모든 데이터를 검증하고 필요한 경우 수정합니다.
        """
        for note in sequence.notes:
            note.velocity = max(0, min(note.velocity, 127))
            note.pitch = max(0, min(note.pitch, 127))
            note.instrument = max(0, min(note.instrument, 15))  # MIDI 채널은 0-15
            note.program = max(0, min(note.program, 127))

        for control_change in sequence.control_changes:
            control_change.control_value = max(0, min(control_change.control_value, 127))
            control_change.control_number = max(0, min(control_change.control_number, 127))
            control_change.instrument = max(0, min(control_change.instrument, 15))

        for pitch_bend in sequence.pitch_bends:
            pitch_bend.bend = max(-8192, min(pitch_bend.bend, 8191))  # 피치 벤드 범위
            pitch_bend.instrument = max(0, min(pitch_bend.instrument, 15))

        return sequence
    
    def log_midi_details(self, sequence):
        self.logger.debug("MIDI Sequence Details:")
        for i, note in enumerate(sequence.notes):
            self.logger.debug(f"Note {i}: pitch={note.pitch}, start_time={note.start_time}, end_time={note.end_time}, velocity={note.velocity}, instrument={note.instrument}, program={note.program}")
        for i, control in enumerate(sequence.control_changes):
            self.logger.debug(f"Control Change {i}: number={control.control_number}, value={control.control_value}, time={control.time}, instrument={control.instrument}")
        for i, tempo in enumerate(sequence.tempos):
            self.logger.debug(f"Tempo {i}: qpm={tempo.qpm}, time={tempo.time}")

    def extend_sequence(self, sequence, target_length):
        """
        주어진 시퀀스를 목표 길이에 맞게 반복 확장합니다.
        """
        original_length = sequence.total_time
        repetitions = int(target_length / original_length) + 1

        extended_sequence = music_pb2.NoteSequence()
        extended_sequence.CopyFrom(sequence)

        for i in range(1, repetitions):
            for note in sequence.notes:
                new_note = extended_sequence.notes.add()
                new_note.CopyFrom(note)
                new_note.start_time += i * original_length
                new_note.end_time += i * original_length

        extended_sequence.total_time = target_length
        return extended_sequence
    
    def validate_sequence(self, sequence):
        if not sequence.notes:
            self.logger.warning("생성된 시퀀스에 음표가 없습니다.")
            return False
        if not sequence.tempos:
            self.logger.warning("생성된 시퀀스에 템포 정보가 없습니다.")
            # 템포 정보 추가
            tempo = sequence.tempos.add()
            tempo.qpm = self.DEFAULT_BPM
            tempo.time = 0
        if sequence.total_time <= 0:
            self.logger.warning("생성된 시퀀스의 총 시간이 0 이하입니다.")
            return False
        if len(sequence.notes) < 2:  # 최소 2개 이상의 음표가 있어야 함
            self.logger.warning("생성된 시퀀스의 음표 수가 너무 적습니다.")
            return False
        return True
    
    def process_groove_output(self, sequence, bpm):
        # groove 모델의 출력을 2마디(32스텝) 단위로 반복
        original_notes = list(sequence.notes)
        new_notes = []
        
        bar_duration = 4 * 60 / bpm  # 4/4 박자 기준, 1마디 길이(초)
        total_duration = 2 * bar_duration  # 2마디
        
        current_time = 0
        while current_time < total_duration:
            for note in original_notes:
                new_note = music_pb2.NoteSequence.Note()
                new_note.CopyFrom(note)
                new_note.start_time += current_time
                new_note.end_time += current_time
                new_notes.append(new_note)
            current_time += bar_duration

        del sequence.notes[:]  # 기존 노트 모두 삭제
        sequence.notes.extend(new_notes)  # 새로운 노트 추가
        sequence.total_time = total_duration

        return sequence
    

# 메인 스크립트 (예시)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = AdvancedCompositionService()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # print("Available genres:", ", ".join(service.GENRES))
        # genre = input("원하는 장르를 입력하세요: ")
        
        # print("Available feels:", ", ".join(service.FEELS))
        # feel = input("원하는 느낌을 입력하세요: ")
        
        # bpm = int(input("BPM을 입력하세요 (60-180): "))
        # genre = 'jazz'
        # feel = 'sad'
        bpm = 220
        
        instruments = {
            'melody': 'Grand piano',
            'bass': 'Standard drum kit',
            'drums': 'Standard drum kit',
            'groove': 'Standard drum kit'
        }
        
        # long_melody 생성
        melody_sequence = service.generate_composition('long_melody', num_outputs=1, length=2000, temperature=0.1, bpm=bpm, min_pitch=48, max_pitch=96, instrument='Grand piano')[0]
        melody_length = melody_sequence.total_time
        melody_midi_path, melody_wav_path = service.save_midi(melody_sequence, os.path.join(output_dir, 'long_melody_composition'), instruments)
        print(f"멜로디 파일 생성: MIDI - {melody_midi_path}, WAV - {melody_wav_path}")

        # # drums 생성
        # drums_sequence = service.generate_composition('drums', num_outputs=1, length=1024, temperature=0.7, bpm=bpm, min_pitch=48, max_pitch=96, instrument='Standard drum kit', genre=genre, feel=feel)[0]
        # drums_midi_path, drums_wav_path = service.save_midi(drums_sequence, os.path.join(output_dir, 'drums_composition'), instruments)
        # print(f"드럼 파일 생성: MIDI - {drums_midi_path}, WAV - {drums_wav_path}")

        # # groove 생성
        # groove_sequence = service.generate_composition('short_melody', num_outputs=1, length=1024, temperature=0.7, bpm=bpm, min_pitch=48, max_pitch=96, instrument='Clarinet', genre=genre, feel=feel)[0]
        # groove_midi_path, groove_wav_path = service.save_midi(groove_sequence, os.path.join(output_dir, 'short_melody_composition'), instruments)
        # print(f"그루브 파일 생성: MIDI - {groove_midi_path}, WAV - {groove_wav_path}")

        # # 모든 시퀀스 결합
        # combined_sequence = service.combine_sequences(melody_sequence, drums_sequence, groove_sequence)

        # MIDI 및 WAV 파일 저장
        # midi_path, wav_path = service.save_midi(combined_sequence, 'backend/output/combined_composition', instruments)
        # print(f"생성된 결합 파일: MIDI - {midi_path}, WAV - {wav_path}")

    except ValueError as ve:
        print(f"값 오류: {str(ve)}")
    except AttributeError as ae:
        print(f"속성 오류: {str(ae)}")
        print("멜로디 객체가 예상된 형식이 아닙니다. 모델 출력을 확인해주세요.")
    except trained_model.NoExtractedExamplesError as e:
        print(f"오류 발생: {str(e)}")
        print("NoteSequence에서 유효한 예제를 추출할 수 없습니다. 입력 데이터를 확인해주세요.")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        if hasattr(e, 'sequence_info'):
            print(f"오류 관련 시퀀스 정보: {e.sequence_info}")
        print("입력 데이터와 모델 설정을 확인해주세요.")