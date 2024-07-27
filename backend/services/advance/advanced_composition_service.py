import os
import logging
import tempfile
import note_seq
import numpy as np
from note_seq.protobuf import music_pb2
from magenta.models.music_vae import configs
from magenta.models.music_vae import TrainedModel
from backend.services.audio_processing_service import AudioProcessingService
from pydub import AudioSegment
from note_seq import sequences_lib
import copy
from dotenv import load_dotenv


class AdvancedCompositionService:
    def __init__(self):
        self.models = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.initialize_models()
        self.audio_service = AudioProcessingService()

        load_dotenv()  # .env 파일에서 환경 변수 로드
        self.MIN_BPM = int(os.getenv('MIN_BPM', 40))
        self.MAX_BPM = int(os.getenv('MAX_BPM', 220))
        self.DEFAULT_BPM = int(os.getenv('DEFAULT_BPM', 120))

    def initialize_models(self):
        model_configs = {
            'short_melody': ('cat-mel_2bar_big', configs.CONFIG_MAP['cat-mel_2bar_big']),
            'long_melody': ('hierdec-mel_16bar', configs.CONFIG_MAP['hierdec-mel_16bar']),
            'trio': ('hierdec-trio_16bar', configs.CONFIG_MAP['hierdec-trio_16bar']),
            'drums': ('nade-drums_2bar_full', configs.CONFIG_MAP['nade-drums_2bar_full']),
            'groove': ('groovae_2bar_humanize', configs.CONFIG_MAP['groovae_2bar_humanize'])
        }

        for model_name, (file_name, config) in model_configs.items():
            try:
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'assets', 'models', file_name)
                
                self.models[model_name] = TrainedModel(
                    config,
                    batch_size=1,
                    checkpoint_dir_or_path=model_path
                )
                self.logger.info(f"{model_name} 모델 초기화 완료")
            except Exception as e:
                self.logger.error(f"{model_name} 모델 초기화 중 오류 발생: {str(e)}")

    def generate_composition(self, composition_type, num_outputs=3, temperature=0.5, length=256, bpm=None):
        if bpm is None:
            bpm = self.DEFAULT_BPM
        bpm = self.validate_bpm(bpm)

        try:
            model = self.models.get(composition_type)
            if model is None:
                raise ValueError(f"모델 '{composition_type}'이(가) 초기화되지 않았거나 사용할 수 없습니다.")

            # 모델별 특성에 따라 길이 조정
            if composition_type in ['short_melody', 'drums', 'groove']:
                adjusted_length = length // 16  # 2마디 단위로 조정
            else:
                adjusted_length = length

            sequences = model.sample(n=num_outputs, length=adjusted_length, temperature=temperature)
            
            valid_sequences = []
            for sequence in sequences:
                if sequence.notes and sequence.total_time > 0:
                    # BPM 설정
                    while sequence.tempos:
                        sequence.tempos.pop()
                    sequence.tempos.add(qpm=bpm)
                    sequence.ticks_per_quarter = 220

                    # 그루브 적용 (드럼 시퀀스인 경우)
                    if composition_type in ['drums', 'groove']:
                        sequence = self.apply_groove(sequence)

                    self.logger.info(f"생성된 {composition_type} 정보: 음표 수: {len(sequence.notes)}, 총 시간: {sequence.total_time}, BPM: {bpm}")
                    valid_sequences.append(sequence)
                else:
                    self.logger.warning(f"빈 {composition_type} 시퀀스가 생성되었습니다. 이를 무시합니다.")
            
            if not valid_sequences:
                raise ValueError(f"유효한 {composition_type} 시퀀스가 생성되지 않았습니다.")
            
            return valid_sequences

        except Exception as e:
            self.logger.error(f"{composition_type} 생성 중 오류 발생: {str(e)}")
            raise

    def apply_groove(self, drum_sequence):
        groove_model = self.models.get('groove')
        if groove_model is None:
            self.logger.warning("그루브 모델이 초기화되지 않았습니다. 그루브를 적용할 수 없습니다.")
            return drum_sequence
        
        try:
            grooved_sequence = groove_model.humanize(drum_sequence)
            return grooved_sequence
        except Exception as e:
            self.logger.error(f"그루브 적용 중 오류 발생: {str(e)}")
            return drum_sequence

    def save_midi(self, sequence, filename, instrument='piano'):
        try:
            midi_path = filename + '.mid'
            wav_path = filename + '.wav'
            mp3_path = filename + '.mp3'
            
            note_seq.sequence_proto_to_midi_file(sequence, midi_path)
            self.logger.info(f"MIDI 파일 저장 완료: {midi_path}")
            
            if not os.path.exists(midi_path):
                raise FileNotFoundError(f"MIDI 파일을 찾을 수 없습니다: {midi_path}")
            
            # WAV 파일로 먼저 변환
            wav_path = self.audio_service.midi_to_wav(midi_path, instrument)
            
            # WAV를 MP3로 변환 (청크 단위로 처리) 
            #--------------------------------------
            # 에러 발생으로 인해 사용 중단 2024-07-24
            #--------------------------------------
            #self.wav_to_mp3(wav_path, mp3_path)
            
            #self.logger.info(f"MP3 파일 생성 완료: {mp3_path}")

            return midi_path, wav_path
        except MemoryError:
            self.logger.error("메모리 부족 오류 발생. 더 작은 크기의 멜로디를 생성해 주세요.")
            raise
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

    def prepare_sequence_for_model(self, sequence, num_steps=256):
        """
        NoteSequence를 모델이 기대하는 형식으로 준비합니다.

        Args:
            sequence (note_seq.NoteSequence): 준비할 NoteSequence
            num_steps (int): 원하는 시퀀스의 길이

        Returns:
            note_seq.NoteSequence: 준비된 NoteSequence
        """
        # 4/4 박자로 가정
        qpm = 120.0
        seconds_per_step = 0.25  # 16분음표 기준
        total_time = num_steps * seconds_per_step

        prepared_sequence = music_pb2.NoteSequence()
        prepared_sequence.tempos.add(qpm=qpm)
        prepared_sequence.ticks_per_quarter = 220

        for note in sequence.notes:
            if note.start_time < total_time:
                new_note = prepared_sequence.notes.add()
                new_note.pitch = note.pitch
                new_note.start_time = note.start_time
                new_note.end_time = min(note.end_time, total_time)
                new_note.velocity = note.velocity
                new_note.instrument = 0  # 모든 노트를 같은 악기로 설정

        prepared_sequence.total_time = min(sequence.total_time, total_time)

        # 시간 시그니처 추가 (4/4 박자 가정)
        time_signature = prepared_sequence.time_signatures.add()
        time_signature.numerator = 4
        time_signature.denominator = 4
        time_signature.time = 0

        self.logger.info(f"전처리된 시퀀스 정보: 음표 수: {len(prepared_sequence.notes)}, "
                        f"총 시간: {prepared_sequence.total_time:.2f}초, "
                        f"ticks_per_quarter: {prepared_sequence.ticks_per_quarter}, "
                        f"템포: {prepared_sequence.tempos[0].qpm}")

        return prepared_sequence
    
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

    def interpolate_melodies(self, start_melody, end_melody, num_steps=5, length=256, bpm=None):
        """
        두 멜로디 사이를 보간하여 새로운 멜로디들을 생성합니다.

        Parameters:
        - start_melody (NoteSequence): 시작 멜로디
        - end_melody (NoteSequence): 끝 멜로디
        - num_steps (int): 보간 단계 수 (기본값: 5)
        - length (int): 생성할 멜로디의 길이 (기본값: 256)
        - bpm (int): 생성할 멜로디의 템포 (기본값: None)

        Returns:
        - list: 보간된 멜로디들의 리스트 (NoteSequence 객체들)

        Raises:
            ValueError: 입력 멜로디가 유효하지 않을 경우
        """
        
        if bpm is None:
            bpm = self.DEFAULT_BPM
        bpm = self.validate_bpm(bpm)

        data_converter = self.models['long_melody']._config.data_converter
        self.logger.info(f"데이터 컨버터 설정: {data_converter.__dict__}")
        self.logger.info(f"시작 멜로디 정보: 음표 수: {len(start_melody.notes)}, 총 시간: {start_melody.total_time:.2f}초")
        self.logger.info(f"끝 멜로디 정보: 음표 수: {len(end_melody.notes)}, 총 시간: {end_melody.total_time:.2f}초")
        if len(start_melody.notes) > 0:
            self.logger.info(f"시작 멜로디 첫 번째 음표: 피치: {start_melody.notes[0].pitch}, 시작 시간: {start_melody.notes[0].start_time}, 종료 시간: {start_melody.notes[0].end_time}")
        if len(end_melody.notes) > 0:
            self.logger.info(f"끝 멜로디 첫 번째 음표: 피치: {end_melody.notes[0].pitch}, 시작 시간: {end_melody.notes[0].start_time}, 종료 시간: {end_melody.notes[0].end_time}")
        try:
            # 입력 멜로디 유효성 검사 및 전처리
            if not isinstance(start_melody, music_pb2.NoteSequence) or not isinstance(end_melody, music_pb2.NoteSequence):
                raise ValueError("시작 또는 끝 멜로디가 올바른 NoteSequence 형식이 아닙니다.")

            if not start_melody.notes or not end_melody.notes:
                raise ValueError("시작 또는 끝 멜로디가 비어 있습니다.")

            # 입력 멜로디 전처리
            start_melody = self.prepare_sequence_for_model(start_melody, length)
            end_melody = self.prepare_sequence_for_model(end_melody, length)

            self.validate_notesequence(start_melody)
            self.validate_notesequence(end_melody)

            # 마디 수 계산 (4/4 박자 가정)
            quarters_per_bar = 4
            start_melody_bars = start_melody.total_time / (60 / start_melody.tempos[0].qpm * quarters_per_bar)
            end_melody_bars = end_melody.total_time / (60 / end_melody.tempos[0].qpm * quarters_per_bar)

            self.logger.info(f"Start melody bars: {start_melody_bars:.2f}")
            self.logger.info(f"End melody bars: {end_melody_bars:.2f}")

            # 마디 수 조정 (필요한 경우)
            target_bars = self.models['long_melody']._config.data_converter.slice_bars
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
                self.logger.info(f"시작 멜로디 인코딩 성공. 형태: {z_start[0].shape if isinstance(z_start, tuple) else z_start.shape}")
            except Exception as e:
                self.logger.error(f"시작 멜로디 인코딩 중 오류 발생: {str(e)}")
                self.logger.exception("인코딩 오류 상세 정보:")
                raise

            try:
                z_end = self.models['long_melody'].encode([end_melody])
                self.logger.info(f"끝 멜로디 인코딩 성공. 형태: {z_end[0].shape if isinstance(z_end, tuple) else z_end.shape}")
            except Exception as e:
                self.logger.error(f"끝 멜로디 인코딩 중 오류 발생: {str(e)}")
                self.logger.exception("인코딩 오류 상세 정보:")
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
                            # BPM 조절
                            adjusted_melody = self.adjust_tempo(melody, bpm)
                            self.logger.info(f"보간된 멜로디 {i+1} 정보: 음표 수: {len(adjusted_melody.notes)}, 총 시간: {adjusted_melody.total_time:.2f}초, BPM: {bpm}")
                            valid_interpolated.append(adjusted_melody)
                        else:
                            self.logger.warning(f"보간된 멜로디 {i+1}에 음표가 없습니다.")
                    else:
                        self.logger.warning(f"보간된 멜로디 {i+1}이(가) 유효한 NoteSequence가 아닙니다. 타입: {type(melody)}")

                except Exception as e:
                    self.logger.error(f"멜로디 {i+1} 디코딩 중 오류 발생: {str(e)}")
                    self.logger.exception("디코딩 오류 상세 정보:")

            if not valid_interpolated:
                raise ValueError("유효한 보간 멜로디가 생성되지 않았습니다.")

            return valid_interpolated

        except Exception as e:
            self.logger.error(f"멜로디 보간 중 오류 발생: {str(e)}")
            self.logger.exception("상세 오류 정보:")
            raise

    def validate_bpm(self, bpm):
        if not (self.MIN_BPM <= bpm <= self.MAX_BPM):
            raise ValueError(f"BPM은 {self.MIN_BPM}에서 {self.MAX_BPM} 사이여야 합니다.")
        return bpm

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

        for note in adjusted_melody.notes:
            note.start_time *= tempo_ratio
            note.end_time *= tempo_ratio

        adjusted_melody.total_time *= tempo_ratio

        return adjusted_melody

# 메인 스크립트 (예시)
if __name__ == "__main__":
    service = AdvancedCompositionService()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # BPM 지정
        bpm = 180  # 예시로 180 BPM 사용
        melodies = service.generate_composition('groove', num_outputs=2, temperature=0.5, length=256, bpm=bpm)
        
        print(f"생성된 멜로디 수: {len(melodies)}")
        for i, melody in enumerate(melodies):
            print(f"멜로디 {i+1} 타입: {type(melody)}")
            if isinstance(melody, music_pb2.NoteSequence):
                print(f"멜로디 {i+1} 음표 수: {len(melody.notes)}, BPM: {melody.tempos[0].qpm}")
            else:
                print(f"멜로디 {i+1}이 예상치 못한 타입입니다.")
        
        if len(melodies) >= 2:
            try:
                interpolated = service.interpolate_melodies(melodies[0], melodies[1], num_steps=5, length=256, bpm=bpm)
                print(f"보간된 멜로디 수: {len(interpolated)}")
                for i, melody in enumerate(interpolated):
                    print(f"보간된 멜로디 {i+1} 타입: {type(melody)}")
                    if isinstance(melody, music_pb2.NoteSequence):
                        print(f"보간된 멜로디 {i+1} 음표 수: {len(melody.notes)}, BPM: {melody.tempos[0].qpm}")
                    else:
                        print(f"보간된 멜로디 {i+1}이 예상치 못한 타입입니다: {type(melody)}")
                    
                    midi_path, wav_path = service.save_midi(
                        melody, 
                        os.path.join(output_dir, f"interpolated_melody_{i}")
                    )
                    print(f"보간된 파일: MIDI - {midi_path}, WAV - {wav_path}")
            except Exception as e:
                print(f"보간 중 오류 발생: {str(e)}")
                print(f"오류 타입: {type(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("보간을 위한 충분한 멜로디가 생성되지 않았습니다.")
    except ValueError as ve:
        print(f"값 오류: {str(ve)}")
    except AttributeError as ae:
        print(f"속성 오류: {str(ae)}")
        print("멜로디 객체가 예상된 형식이 아닙니다. 모델 출력을 확인해주세요.")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()