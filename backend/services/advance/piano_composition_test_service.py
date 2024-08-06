import os
import logging
import tensorflow as tf
from note_seq.protobuf import generator_pb2
import note_seq
from magenta.models.shared import sequence_generator_bundle
from magenta.models.performance_rnn import performance_sequence_generator
from magenta.models.performance_rnn import performance_model

class PianoCompositionService:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.initialize_model()

    def initialize_model(self):
        """Performance RNN 모델을 초기화합니다."""
        self.logger.debug("모델 초기화 시작")
        
        bundle_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'performance_with_dynamics.mag')
        
        bundle = sequence_generator_bundle.read_bundle_file(bundle_file)
        config_id = bundle.generator_details.id
        config = performance_model.default_configs[config_id]
        
        self.generator = performance_sequence_generator.PerformanceRnnSequenceGenerator(
            model=performance_model.PerformanceRnnModel(config),
            details=bundle.generator_details,
            steps_per_second=config.steps_per_second,
            num_velocity_bins=config.num_velocity_bins,
            bundle=bundle)

        self.generator.initialize()
        self.logger.debug("모델 초기화 완료")

    def generate_music(self, total_time=120, temperature=1.0):
        """피아노 음악을 생성합니다."""
        self.logger.debug("음악 생성 시작")
        
        generator_options = generator_pb2.GeneratorOptions()  # 이 줄을 수정했습니다
        generator_options.args['temperature'].float_value = temperature
        generator_options.generate_sections.add(
            start_time=0,
            end_time=total_time
        )
        
        sequence = self.generator.generate(generator_options)
        
        self.logger.debug("음악 생성 완료")
        return sequence

    def save_midi(self, sequence, filename):
        """MIDI 시퀀스를 파일로 저장합니다."""
        self.logger.debug(f"MIDI 파일 저장 시작: {filename}")
        note_seq.sequence_proto_to_midi_file(sequence, filename)
        self.logger.debug(f"MIDI 파일 저장 완료: {filename}")
        return filename

    def generate_and_save_music(self, filename, duration=120, temperature=1.0):
        """음악을 생성하고 저장합니다."""
        self.logger.info("음악 생성 및 저장 시작")
        try:
            sequence = self.generate_music(duration, temperature)
            midi_path = self.save_midi(sequence, filename)
            self.logger.info(f"음악 생성 및 저장 완료: {midi_path}")
            return midi_path
        except Exception as e:
            self.logger.error(f"음악 생성 및 저장 중 오류 발생: {str(e)}")
            raise

# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    piano_service = PianoCompositionService()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        midi_path = piano_service.generate_and_save_music(
            os.path.join(output_dir, 'generated_piano_music.mid'),
            duration=120,
            temperature=1.0
        )
        print(f"생성된 MIDI 파일: {midi_path}")
    except Exception as e:
        print(f"오류 발생: {str(e)}")