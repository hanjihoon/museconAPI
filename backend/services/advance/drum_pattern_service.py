from magenta.models.drums_rnn import drums_rnn_sequence_generator
from note_seq.protobuf import music_pb2

class DrumPatternService:
    def init(self):
        self.drum_model = drums_rnn_sequence_generator.get_generator_map()['drums_rnn']

    def generate_drum_pattern(self, genre, length):
        generator_options = music_pb2.GeneratorOptions()
        generator_options.args['temperature'].float_value = 1.0
        generator_options.args['genre'].string_value = genre
        generator_options.generate_sections.add(
            start_time=0,
            end_time=length
        )

        sequence = self.drum_model.generate(generator_options)
        return sequence  # 드럼 패턴은 이미 NoteSequence 형태