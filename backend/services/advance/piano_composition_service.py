import os
import logging
import threading
import tensorflow as tf
from memory_profiler import profile
from note_seq.protobuf import music_pb2, generator_pb2
from magenta.models.shared import sequence_generator_bundle
from magenta.models.performance_rnn import performance_model, performance_sequence_generator
import note_seq
from magenta.contrib import training as contrib_training
from backend.services.audio_processing_service import AudioProcessingService
import inspect

# TensorFlow 설정 및 GPU 확인
tf.get_logger().setLevel('DEBUG')
print("TensorFlow 버전:", tf.__version__)
print("GPU 사용 가능:", tf.test.is_built_with_cuda())
print("사용 가능한 GPU:", tf.config.list_physical_devices('GPU'))

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=300):
    """
    주어진 함수를 지정된 시간 내에 실행하는 함수입니다.
    
    :param func: 실행할 함수
    :param args: 함수에 전달할 위치 인자
    :param kwargs: 함수에 전달할 키워드 인자
    :param timeout_duration: 타임아웃 시간 (초)
    :return: 함수의 실행 결과
    :raises TimeoutError: 함수가 지정된 시간 내에 완료되지 않을 경우
    :raises Exception: 함수 실행 중 발생한 기타 예외
    """
    result = [None]
    exception = [None]
    
    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout_duration)
    
    if thread.is_alive():
        raise TimeoutError(f"함수 호출이 {timeout_duration}초 후 타임아웃되었습니다.")
    if exception[0]:
        raise exception[0]
    return result[0]

class PianoCompositionService:
    """
    피아노 작곡 서비스를 제공하는 클래스입니다.
    """

    def __init__(self):
        """
        PianoCompositionService 클래스의 생성자입니다.
        로거를 설정하고 Performance RNN 모델을 초기화합니다.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.initialize_performance_rnn()
        self.audio_service = AudioProcessingService()

    def initialize_performance_rnn(self):
        self.logger.debug("Performance RNN 모델 초기화 시작")
        bundle_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'assets', 'models', 'performance_with_dynamics.mag.mag')
        
        try:
            bundle = sequence_generator_bundle.read_bundle_file(bundle_file)
            self.logger.info(f"번들 파일 로드 성공: {bundle_file}")
            
            # 번들 구조 로깅
            self.logger.debug(f"Bundle attributes: {dir(bundle)}")
            self.logger.debug(f"Generator details attributes: {dir(bundle.generator_details)}")
            self.logger.debug(f"Generator details: {bundle.generator_details}")
            
            # hparams와 num_velocity_bins 추출 시도
            hparams = contrib_training.HParams()
            if hasattr(bundle, 'hparams'):
                hparams.parse(bundle.hparams)
            elif hasattr(bundle.generator_details, 'hparams'):
                hparams.parse(bundle.generator_details.hparams)
            else:
                self.logger.warning("hparams를 찾을 수 없습니다. 기본값을 사용합니다.")
            
            num_velocity_bins = getattr(bundle.generator_details, 'num_velocity_bins', 0)
            
            self.logger.debug(f"Extracted hparams: {hparams}")
            self.logger.debug(f"Extracted num_velocity_bins: {num_velocity_bins}")
            
            # PerformanceRnnConfig 직접 사용
            config = performance_model.PerformanceRnnConfig(
                hparams=hparams,
                details=bundle.generator_details,
                encoder_decoder=note_seq.PerformanceOneHotEncoding(num_velocity_bins=num_velocity_bins)
            )
            
            self.generator = performance_sequence_generator.PerformanceRnnSequenceGenerator(
                model=performance_model.PerformanceRnnModel(config),
                details=bundle.generator_details,
                steps_per_second=getattr(config, 'steps_per_second', 100),  # 기본값 설정
                num_velocity_bins=num_velocity_bins,
                bundle=bundle)
            
            self.logger.debug(f"Generator initialized: {self.generator}")
            self.logger.debug(f"Generator type: {type(self.generator)}")
            self.logger.debug(f"Generator attributes: {dir(self.generator)}")

            # generate 메서드 래핑
            def wrapped_generate(generator_options):
                self.logger.debug("wrapped_generate 함수 호출됨")
                self.logger.debug(f"generator_options 타입: {type(generator_options)}")
                self.logger.debug(f"generator_options 내용: {generator_options}")
                self.logger.debug(f"self.generator 타입: {type(self.generator)}")
                self.logger.debug(f"self.generator 속성: {dir(self.generator)}")
                
                # generator_options에서 필요한 값들을 추출
                temperature = generator_options.args['temperature'].float_value
                
                pitch_class_histogram = None
                if 'pitch_class_histogram' in generator_options.args:
                    pch = generator_options.args['pitch_class_histogram']
                    self.logger.debug(f"pch 타입: {type(pch)}")
                    self.logger.debug(f"pch 속성: {dir(pch)}")
                    if hasattr(pch, 'float_value'):
                        pitch_class_histogram = pch.float_value
                
                note_density = None
                if 'note_density' in generator_options.args:
                    note_density = generator_options.args['note_density'].float_value
                
                self.logger.debug(f"추출된 값들: temperature={temperature}, pitch_class_histogram={pitch_class_histogram}, note_density={note_density}")
                
                try:
                    # generate 메서드 직접 호출
                    result = self.generator.generate(generator_options)
                    self.logger.debug("generate 메서드 성공적으로 완료")
                    return result
                except Exception as e:
                    self.logger.error(f"generate 메서드 내부에서 오류 발생: {str(e)}")
                    self.logger.exception("상세 오류 정보:")
                    raise


            self.generator.generate = wrapped_generate
            # PerformanceRnnSequenceGenerator 클래스의 generate 메서드 확인
            self.logger.debug(f"PerformanceRnnSequenceGenerator.generate 메서드: {inspect.getsource(performance_sequence_generator.PerformanceRnnSequenceGenerator.generate)}")

            try:
                self.generator.initialize()
                self.logger.info("Performance RNN 모델 초기화 완료")
            except Exception as e:
                self.logger.error(f"모델 초기화 중 오류 발생: {str(e)}")
                self.logger.exception("상세 오류 정보:")
                raise
        except Exception as e:
            self.logger.error(f"모델 초기화 중 오류 발생: {str(e)}")
            self.logger.exception("상세 오류 정보:")
            raise

    @profile
    def generate_piano_music(self, duration, temperature, pitch_class_histogram, note_density):
        """
        피아노 음악을 생성하는 메서드입니다.
        
        :param duration: 생성할 음악의 총 길이 (초)
        :param temperature: 생성의 무작위성을 제어하는 온도 매개변수
        :param pitch_class_histogram: 음높이 클래스 분포
        :param note_density: 노트 밀도
        :return: 생성된 음악 시퀀스
        """
        self.logger.debug("generate_piano_music 메서드 시작")
        self.logger.debug(f"입력 값: duration={duration}, temperature={temperature}, pitch_class_histogram={pitch_class_histogram}, note_density={note_density}")
        
        sequence = music_pb2.NoteSequence()
        
        generator_options = generator_pb2.GeneratorOptions()
        generator_options.args['temperature'].float_value = temperature
        generator_options.generate_sections.add(
            start_time=0,
            end_time=duration
        )
        
        if pitch_class_histogram is not None:
            pch_arg = generator_options.args['pitch_class_histogram']
            self.logger.debug(f"pch_arg 타입: {type(pch_arg)}")
            self.logger.debug(f"pch_arg 속성: {dir(pch_arg)}")
            
            for value in pitch_class_histogram:
                pch_arg.float_values.append(value)
        
        if note_density is not None:
            generator_options.args['note_density'].float_value = note_density
        
        self.logger.debug(f"생성된 generator_options: {generator_options}")
        
        try:
            generated_sequence = self.generator.generate(generator_options)
            self.logger.debug(f"생성된 시퀀스 타입: {type(generated_sequence)}")
            if isinstance(generated_sequence, music_pb2.NoteSequence):
                sequence.CopyFrom(generated_sequence)
            else:
                sequence.ParseFromString(generated_sequence)
            self.logger.debug("시퀀스 생성 완료")
        except Exception as e:
            self.logger.error(f"시퀀스 생성 중 오류 발생: {str(e)}")
            self.logger.exception("상세 오류 정보:")
            raise
        
        self.logger.debug("generate_piano_music 메서드 완료")
        return sequence

    def _create_generator_options(self, start_time, end_time, temperature, pitch_class_histogram, note_density):
        """
        generator_options를 생성하는 내부 메서드입니다.
        
        :param start_time: 시작 시간
        :param end_time: 종료 시간
        :param temperature: 온도 매개변수
        :param pitch_class_histogram: 음높이 클래스 분포
        :param note_density: 노트 밀도
        :return: 설정된 generator_options 객체
        """
        generator_options = generator_pb2.GeneratorOptions()
        generator_options.args['temperature'].float_value = temperature
        generator_options.generate_sections.add(
            start_time=start_time,
            end_time=end_time
        )
        
        if pitch_class_histogram is not None:
            pitch_class_histogram_arg = generator_options.args['pitch_class_histogram']
            if isinstance(pitch_class_histogram, list):
                for value in pitch_class_histogram:
                    pitch_class_histogram_arg.float_values.append(value)
            else:
                pitch_class_histogram_arg.float_value = pitch_class_histogram
        
        if note_density is not None:
            generator_options.args['note_density'].float_value = note_density
        
        self.logger.debug(f"Created generator_options: {generator_options}")
        return generator_options

    def save_piano_music(self, sequence, filename):
        """
        생성된 피아노 음악을 MIDI 파일로 저장하는 메서드입니다.
        
        :param sequence: 저장할 음악 시퀀스
        :param filename: 저장할 파일 이름 (확장자 제외)
        :return: 저장된 MIDI 파일의 경로
        """
        midi_path = filename + '.mid'
        note_seq.sequence_proto_to_midi_file(sequence, midi_path)
        self.logger.info(f"피아노 음악 MIDI 파일 저장 완료: {midi_path}")
        return midi_path

    def generate_and_save_piano_music(self, filename, duration=30, temperature=1.0, pitch_class_histogram=None, note_density=None):
        """
        피아노 음악을 생성하고 저장하는 메서드입니다.
        
        :param filename: 저장할 파일 이름 (확장자 제외)
        :param duration: 생성할 음악의 길이 (초)
        :param temperature: 생성의 무작위성을 제어하는 온도 매개변수
        :param pitch_class_histogram: 음높이 클래스 분포
        :param note_density: 노트 밀도
        :return: 생성된 MIDI 파일과 WAV 파일의 경로
        """
        self.logger.info("음악 생성 시작")
        try:
            if pitch_class_histogram is None or not isinstance(pitch_class_histogram, list) or len(pitch_class_histogram) != 12:
                self.logger.warning("pitch_class_histogram이 올바르지 않습니다. 기본값으로 설정합니다.")
                pitch_class_histogram = [1.0] * 12  # 모든 음높이 클래스에 동일한 확률 부여

            self.logger.debug(f"사용될 pitch_class_histogram: {pitch_class_histogram}")

            sequence = self.generate_piano_music(duration, temperature, pitch_class_histogram, note_density)
            self.logger.info("MIDI 파일 저장 시작")
            midi_path = self.save_piano_music(sequence, filename)
            self.logger.info(f"MIDI 파일 저장 완료: {midi_path}")
            
            wav_path = f"{filename}.wav"
            
            instruments = {
                'piano': 'Grand piano',
            }
            
            self.logger.info("WAV 파일 생성 시작")
            self.audio_service.midi_to_wav(midi_path, wav_path, instruments)
            self.logger.info(f"WAV 파일 생성 완료: {wav_path}")
            
            return midi_path, wav_path
        except Exception as e:
            self.logger.error(f"음악 생성 및 저장 중 오류 발생: {str(e)}")
            raise

# 메인 실행 부분
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    piano_service = PianoCompositionService()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("피아노 음악 생성을 시작합니다.")
        
        # temperature = float(input("Temperature를 입력하세요 (0.1-1.0, 기본값 1.0): ") or 1.0)
        # duration = int(input("생성할 음악의 길이를 입력하세요 (초 단위, 기본값 30): ") or 30)
        temperature = 1.0
        duration = 30
        
        # use_conditions = input("피치 클래스와 노트 밀도를 설정하시겠습니까? (y/n): ").lower() == 'y'
        use_conditions = 'y'
        pitch_class_histogram = None
        note_density = None
        
        if use_conditions:
            # pitch_class_histogram_input = input("피치 클래스 히스토그램을 입력하세요 (12개의 숫자, 쉼표로 구분): ")
            pitch_class_histogram_input = '10,10,10,10,10,10,10,10,10,10,5,5'
            pitch_class_histogram = [float(x) for x in pitch_class_histogram_input.split(',')]
            if len(pitch_class_histogram) != 12:
                raise ValueError("피치 클래스 히스토그램은 정확히 12개의 값을 가져야 합니다.")
            # note_density = float(input("노트 밀도를 입력하세요 (0.0-2.0): "))
            note_density = 2.0

        midi_path, wav_path = piano_service.generate_and_save_piano_music(
            os.path.join(output_dir, 'generated_piano_music'),
            duration=duration,
            temperature=temperature,
            pitch_class_histogram=pitch_class_histogram,
            note_density=note_density
        )
        
        print(f"생성된 피아노 음악 파일:")
        print(f"MIDI: {midi_path}")
        print(f"WAV: {wav_path}")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        print("상세 오류 정보:")
        import traceback
        traceback.print_exc()