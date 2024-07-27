
import random
from note_seq.protobuf import music_pb2

class BasslineService:
    def generate_bassline(self, chord_progression, genre):
        bassline = music_pb2.NoteSequence()
        for i, chord in enumerate(chord_progression):
            root_note = chord[0]  # 화음의 근음
            bass_note = music_pb2.NoteSequence.Note()
            bass_note.pitch = root_note - 12  # 한 옥타브 아래
            bass_note.start_time = i
            bass_note.end_time = i + 0.9  # 약간의 공백 추가
            bass_note.velocity = 80 + random.randint(-10, 10)  # 약간의 다이나믹 추가
            bassline.notes.extend([bass_note])
        return bassline