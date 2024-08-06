from note_seq.protobuf import music_pb2
import music21
from backend.services.advance.genre_styles import *
from backend.services.advance.feel_modifiers import *

class MusicGenerator:
    GENRES = ["classical", "jazz", "rock", "pop", "electronic", "hip_hop", "r_and_b", "country", "folk", "latin"]
    FEELS = ["happy", "sad", "energetic", "calm", "mysterious", "romantic", "nostalgic", "epic", "playful", "dark"]

    @staticmethod
    def apply_genre_and_feel(sequence, genre, feel):
        """
        NoteSequence에 지정된 장르와 느낌을 적용합니다.
        
        Parameters:
        - sequence (music_pb2.NoteSequence): 원본 음악 시퀀스
        - genre (str): 적용할 장르 (예: 'jazz', 'rock')
        - feel (str): 적용할 느낌 (예: 'happy', 'sad')
        
        Returns:
        - music_pb2.NoteSequence: 장르와 느낌이 적용된 수정된 시퀀스
        """
        logger = logging.getLogger('MusicGenerator')
        logger.info(f"시퀀스에 '{genre}' 장르와 '{feel}' 느낌 적용 시작")
        
        stream = MusicGenerator.notesequence_to_stream(sequence)
        logger.info(f"NoteSequence를 music21 스트림으로 변환 완료. 스트림 길이: {len(stream)}")
        
        genre_function = globals().get(f"apply_{genre}_style")
        if genre_function:
            logger.info(f"{genre} 스타일 적용 시작")
            try:
                stream = genre_function(stream)
                logger.info(f"{genre} 스타일 적용 완료")
            except Exception as e:
                logger.error(f"{genre} 스타일 적용 중 오류 발생: {str(e)}")
        else:
            logger.warning(f"{genre} 스타일에 대한 함수를 찾을 수 없음")
        
        feel_function = globals().get(f"apply_{feel}_feel")
        if feel_function:
            logger.info(f"{feel} 느낌 적용 시작")
            try:
                stream = feel_function(stream)
                logger.info(f"{feel} 느낌 적용 완료")
            except Exception as e:
                logger.error(f"{feel} 느낌 적용 중 오류 발생: {str(e)}")
        else:
            logger.warning(f"{feel} 느낌에 대한 함수를 찾을 수 없음")
        
        modified_sequence = MusicGenerator.stream_to_notesequence(stream)
        logger.info(f"수정된 스트림을 NoteSequence로 변환 완료. 음표 수: {len(modified_sequence.notes)}")
        
        return modified_sequence

    @staticmethod
    def notesequence_to_stream(sequence):
        stream = music21.stream.Stream()
        for note in sequence.notes:
            n = music21.note.Note(note.pitch)
            n.quarterLength = (note.end_time - note.start_time) / 0.5
            n.volume.velocity = note.velocity
            stream.insert(note.start_time, n)
        return stream

    @staticmethod
    def stream_to_notesequence(stream):
        sequence = music_pb2.NoteSequence()
        for element in stream.recurse().notesAndRests:
            if isinstance(element, music21.note.Note):
                note = sequence.notes.add()
                note.pitch = element.pitch.midi
                note.start_time = element.offset * 0.5
                note.end_time = note.start_time + (element.duration.quarterLength * 0.5)
                note.velocity = int(element.volume.velocity)
        return sequence