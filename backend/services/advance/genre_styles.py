import music21
import random
import logging

def apply_classical_style(stream):
    # 클래식 스타일 구현
    key = stream.analyze('key')
    chordify = stream.chordify()
    for chord in chordify.recurse().getElementsByClass('Chord'):
        if random.random() < 0.3:
            new_chord = create_classical_chord(chord, key)
            stream.insertIntoNoteOrChord(chord.offset, new_chord)
    return stream

def apply_jazz_style(stream):
    """
    주어진 music21 스트림에 재즈 스타일을 적용합니다.
    
    이 함수는 다음과 같은 재즈 특성을 적용합니다:
    1. 스윙 리듬: 홀수 박자의 음표를 늘리고 짝수 박자의 음표를 줄입니다.
    2. 7th와 9th 코드: 기존 코드에 7th와 9th 음을 추가합니다.
    3. 블루 노트: 일부 음표를 블루 노트로 변경합니다.
    
    Parameters:
    - stream (music21.stream.Stream): 수정할 음악 스트림
    
    Returns:
    - music21.stream.Stream: 재즈 스타일이 적용된 수정된 스트림
    """
    logger = logging.getLogger('GenreStyle')
    logger.info("재즈 스타일 적용 시작")
    
    # 드럼 트랙 감지 (MIDI 채널 10 또는 프로그램 0을 사용하는 경우)
    is_drum_track = any(
        (isinstance(element, music21.instrument.Instrument) and element.midiChannel == 10) or
        (hasattr(element, 'isPercussion') and element.isPercussion) or
        (isinstance(element, music21.instrument.UnpitchedPercussion))
        for element in stream.recurse()
    )
    
    if is_drum_track:
        logger.info("드럼 트랙 감지, 재즈 드럼 패턴 적용")
        stream = apply_jazz_drum_pattern(stream)
    else:
        # 기존의 재즈 스타일 적용 (멜로디, 코드 등에 대해)
        logger.info("멜로디/화성 트랙 감지, 일반 재즈 스타일 적용")
        # 스윙 리듬 적용
        swing_notes = 0
        for note in stream.recurse().notesAndRests:
            if isinstance(note, music21.note.Note):
                if note.beat % 2 == 1:  # 홀수 박자
                    note.duration.quarterLength *= 1.5
                    swing_notes += 1
                else:  # 짝수 박자
                    note.duration.quarterLength *= 0.5
                    swing_notes += 1
        logger.info(f"스윙 리듬 적용 완료: {swing_notes}개의 음표 수정됨")

        # 7th, 9th 코드 추가
        chordify = stream.chordify()
        chord_count = 0
        for chord in chordify.recurse().getElementsByClass('Chord'):
            if random.random() < 0.6:  # 60% 확률로 7th 추가
                seventh = chord.root().transpose('M7')
                chord.add(seventh)
                chord_count += 1
                if random.random() < 0.3:  # 30% 확률로 9th 추가
                    ninth = chord.root().transpose('M9')
                    chord.add(ninth)
                    chord_count += 1
        logger.info(f"7th와 9th 코드 추가 완료: {chord_count}개의 코드 수정됨")

        # 블루 노트 추가
        blue_notes = 0
        for note in stream.recurse().notes:
            if isinstance(note, music21.note.Note) and random.random() < 0.1:  # 10% 확률로 블루 노트 추가
                blue_note = note.transpose(music21.interval.Interval('m3'))
                stream.insert(note.offset, blue_note)
                blue_notes += 1
        logger.info(f"블루 노트 추가 완료: {blue_notes}개의 블루 노트 추가됨")
    
    logger.info("재즈 스타일 적용 완료")
    return stream


def apply_rock_style(self, stream):
        """
        록 스타일을 적용합니다.
        """
        # 파워 코드 추가
        for note in stream.recurse().notes:
            if random.random() < 0.3:  # 30% 확률로 파워 코드 추가
                fifth = note.transpose('P5')
                stream.insert(note.offset, fifth)

        # 강한 리듬 패턴 적용
        for measure in stream.getElementsByClass('Measure'):
            if random.random() < 0.5:  # 50% 확률로 강한 리듬 패턴 적용
                self.apply_rock_rhythm(measure)

def apply_rock_rhythm(self, measure):
    """
    록 스타일의 리듬 패턴을 적용합니다.
    """
    rhythm = [1, 0.5, 0.5, 1, 1]  # 예: 4분음표, 8분음표, 8분음표, 4분음표, 4분음표
    for i, note in enumerate(measure.notesAndRests):
        note.quarterLength = rhythm[i % len(rhythm)]

def apply_pop_style(self, stream):
    # 4코드 진행 적용
    chords = ['I', 'V', 'vi', 'IV']
    chord_progression = music21.roman.RomanNumeral(chords[0], stream.analyze('key'))
    for i, measure in enumerate(stream.getElementsByClass('Measure')):
        chord = music21.roman.RomanNumeral(chords[i % len(chords)], stream.analyze('key'))
        measure.insert(0, chord)

def apply_electronic_style(self, stream):
    # 아르페지오 패턴 추가
    for measure in stream.getElementsByClass('Measure'):
        if random.random() < 0.5:
            self.add_arpeggio_pattern(measure)

def apply_hip_hop_style(self, stream):
    # 비트 강조
    for note in stream.recurse().notesAndRests:
        if note.beat == 1 or note.beat == 3:
            note.volume.velocity = min(100, note.volume.velocity * 1.2)

def apply_r_and_b_style(self, stream):
    # 7th와 9th 코드 추가
    for chord in stream.recurse().getElementsByClass('Chord'):
        if random.random() < 0.4:
            chord.add(chord.root().transpose('M7'))
            if random.random() < 0.3:
                chord.add(chord.root().transpose('M9'))

def apply_country_style(self, stream):
    # 기타 피킹 패턴 추가
    for measure in stream.getElementsByClass('Measure'):
        if random.random() < 0.5:
            self.add_guitar_picking_pattern(measure)

def apply_folk_style(self, stream):
    # 간단한 화성 구조와 반복 적용
    key = stream.analyze('key')
    chords = [key.tonic, key.subdominant, key.dominant]
    for i, measure in enumerate(stream.getElementsByClass('Measure')):
        chord = music21.chord.Chord(chords[i % len(chords)])
        measure.insert(0, chord)

def apply_latin_style(self, stream):
    # 라틴 리듬 패턴 적용
    rhythm = [1, 0.5, 0.5, 1, 0.5, 0.5]
    for i, note in enumerate(stream.recurse().notesAndRests):
        note.quarterLength = rhythm[i % len(rhythm)]

############################################################################################
######### 핼퍼 함수
############################################################################################

def create_classical_chord(chord, key):
    root = chord.root()
    third = key.getScaleDegreeFromPitch(root) + 2
    fifth = key.getScaleDegreeFromPitch(root) + 4
    new_chord = music21.chord.Chord([root, third, fifth])
    new_chord.quarterLength = chord.quarterLength
    return new_chord

def apply_jazz_drum_pattern(stream):
    logger = logging.getLogger('GenreStyle')
    # 기본적인 재즈 드럼 패턴 정의
    jazz_pattern = [
        {'pitch': 42, 'duration': 1.0, 'velocity': 100},  # 하이햇 (닫힘)
        {'pitch': 38, 'duration': 1.0, 'velocity': 80},   # 스네어
        {'pitch': 42, 'duration': 1.0, 'velocity': 60},   # 하이햇 (닫힘)
        {'pitch': 36, 'duration': 1.0, 'velocity': 100},  # 베이스 드럼
    ]
    
    new_stream = music21.stream.Stream()
    measure_duration = 4.0  # 4/4 박자 가정
    current_time = 0.0
    
    while current_time < stream.duration.quarterLength:
        for note_info in jazz_pattern:
            new_note = music21.note.Note(note_info['pitch'])
            new_note.duration.quarterLength = note_info['duration']
            new_note.volume.velocity = note_info['velocity']
            new_stream.insert(current_time, new_note)
        current_time += measure_duration
    
    logger.info(f"재즈 드럼 패턴 적용 완료: {len(new_stream.recurse().notes)}개의 새 드럼 노트 생성")
    return new_stream