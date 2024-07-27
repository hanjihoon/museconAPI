from magenta.models.melody_rnn import melody_rnn_sequence_generator
from note_seq.protobuf import music_pb2

class ChordProgressionService:
    def init(self):
        self.chord_model = melody_rnn_sequence_generator.get_generator_map()['chord_pitches_improv']

    def generate_chord_progression(self, genre, length):
        generator_options = music_pb2.GeneratorOptions()
        generator_options.args['temperature'].float_value = 1.0
        generator_options.args['genre'].string_value = genre
        generator_options.generate_sections.add(
            start_time=0,
            end_time=length
        )

        sequence = self.chord_model.generate(generator_options)
        return self._extract_chords(sequence)

    def _extract_chords(self, sequence):
        # 시퀀스에서 화음 추출 로직
        chords = []
        # ... 화음 추출 로직 구현 ...
        return chords