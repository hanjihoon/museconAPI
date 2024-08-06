import music21
import random
import logging

def apply_happy_feel(stream):
    key = stream.analyze('key')
    if key.mode == 'minor':
        stream.transpose(music21.interval.Interval(key.tonic, key.relative.tonic))
    
    for tempo in stream.recurse().getElementsByClass('MetronomeMark'):
        tempo.number *= 1.1
    
    for note in stream.recurse().notes:
        if random.random() < 0.4:
            note.articulations.append(music21.articulations.Staccato())
    return stream

def apply_sad_feel(stream):
    """
    주어진 music21 스트림에 슬픈 느낌을 적용합니다.
    
    이 함수는 다음과 같은 특성을 적용하여 슬픈 느낌을 만듭니다:
    1. 템포 감소: 전체적인 템포를 줄여 느린 분위기를 만듭니다.
    2. 음표 길이 증가: 음표의 지속 시간을 늘려 레가토 효과를 만듭니다.
    3. 다이나믹 감소: 음량을 줄여 부드러운 느낌을 줍니다.
    4. 낮은 음역대 강조: 높은 음역대의 음표를 낮은 음역대로 이동시킵니다.
    
    Parameters:
    - stream (music21.stream.Stream): 수정할 음악 스트림
    
    Returns:
    - music21.stream.Stream: 슬픈 느낌이 적용된 수정된 스트림
    """
    logger = logging.getLogger('FeelModifier')
    logger.info("슬픈 느낌 적용 시작")
    
    # 템포 감소
    tempo_change = False
    for tempo in stream.recurse().getElementsByClass('MetronomeMark'):
        original_tempo = tempo.number
        tempo.number *= 0.4  # 60% 느리게
        tempo_change = True
        logger.info(f"템포 조절: {original_tempo} BPM에서 {tempo.number} BPM으로 변경")
    
    if not tempo_change:
        # 템포 정보가 없는 경우 새로 추가
        new_tempo = music21.tempo.MetronomeMark(number=60)  # 60 BPM으로 설정
        stream.insert(0, new_tempo)
        logger.info(f"새로운 템포 추가: 60 BPM")
    
    # 음표 길이 증가 및 다이나믹 감소
    modified_notes = 0
    for note in stream.recurse().notes:
        note.duration.quarterLength *= 2.0  # 100% 길게
        note.volume.velocity = max(20, min(note.volume.velocity * 0.6, 60))  # 더 조용하게
        modified_notes += 1
    logger.info(f"음표 수정 완료: {modified_notes}개의 음표 길이 증가 및 음량 감소")
    
    # 낮은 음역대 강조 (드럼 트랙 제외)
    lowered_notes = 0
    if not any(isinstance(inst, music21.instrument.UnpitchedPercussion) for inst in stream.recurse().getElementsByClass('Instrument')):
        for note in stream.recurse().notes:
            if isinstance(note, music21.note.Note) and note.pitch.midi > 72:  # C5 이상의 높은 음
                note.transpose('P-8', inPlace=True)  # 옥타브 아래로
                lowered_notes += 1
    logger.info(f"낮은 음역대 강조 완료: {lowered_notes}개의 음표를 낮은 음역대로 이동")

    logger.info("슬픈 느낌 적용 완료")
    return stream

def apply_energetic_feel(self, stream):
    # 템포 증가 및 스타카토 적용
    for tempo in stream.recurse().getElementsByClass('MetronomeMark'):
        tempo.number *= 1.2
    for note in stream.recurse().notes:
        if random.random() < 0.6:
            note.articulations.append(music21.articulations.Staccato())

def apply_calm_feel(self, stream):
    # 템포 감소 및 다이나믹 조정
    for tempo in stream.recurse().getElementsByClass('MetronomeMark'):
        tempo.number *= 0.8
    for note in stream.recurse().notes:
        note.volume.velocity = min(80, note.volume.velocity)

def apply_mysterious_feel(self, stream):
    # 불협화음 추가 및 템포 변화
    for chord in stream.recurse().getElementsByClass('Chord'):
        if random.random() < 0.3:
            chord.add(chord.root().transpose('A4'))
    for tempo in stream.recurse().getElementsByClass('MetronomeMark'):
        tempo.number *= random.uniform(0.9, 1.1)

def apply_romantic_feel(self, stream):
    # 레가토 및 루바토 적용
    for note in stream.recurse().notes:
        note.articulations.append(music21.articulations.Legato())
    for tempo in stream.recurse().getElementsByClass('MetronomeMark'):
        tempo.number *= random.uniform(0.95, 1.05)

def apply_nostalgic_feel(self, stream):
    # 마이너 키로 변환 및 템포 감소
    key = stream.analyze('key')
    if key.mode == 'major':
        stream.transpose(music21.interval.Interval(key.tonic, key.parallel.tonic))
    for tempo in stream.recurse().getElementsByClass('MetronomeMark'):
        tempo.number *= 0.9

def apply_epic_feel(self, stream):
    # 옥타브 더블링 및 다이나믹 증가
    for note in stream.recurse().notes:
        octave = note.transpose('P8')
        stream.insert(note.offset, octave)
        note.volume.velocity = min(127, note.volume.velocity * 1.3)

def apply_playful_feel(self, stream):
    # 스타카토 및 악센트 추가
    for note in stream.recurse().notes:
        if random.random() < 0.4:
            note.articulations.append(music21.articulations.Staccato())
        elif random.random() < 0.3:
            note.articulations.append(music21.articulations.Accent())

def apply_dark_feel(self, stream):
    # 저음역대 강조 및 불협화음 추가
    for note in stream.recurse().notes:
        if note.pitch.midi > 60:
            note.transpose('P-8', inPlace=True)
    for chord in stream.recurse().getElementsByClass('Chord'):
        if random.random() < 0.3:
            chord.add(chord.root().transpose('d5'))
