import logging
from note_seq.protobuf import music_pb2
import os
from backend.services.audio_enhancement_service import AudioEnhancementService
from backend.services.audio_processing_service import AudioProcessingService
from backend.services.audio_cleaning_service import AudioCleaningService
from pydub import AudioSegment
from note_seq import sequences_lib

class CompositionEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.audio_service = AudioProcessingService()
        self.enhancement_service = AudioEnhancementService()
        self.cleaning_service = AudioCleaningService()

    def analyze_melody(self, melody):
        notes = [note.pitch for note in melody.notes]
        durations = [note.end_time - note.start_time for note in melody.notes]
        if not notes:
            return {'avg_pitch': 0, 'pitch_range': 0, 'avg_duration': 0, 'total_duration': 0}
        return {
            'avg_pitch': sum(notes) / len(notes),
            'pitch_range': max(notes) - min(notes),
            'avg_duration': sum(durations) / len(durations),
            'total_duration': melody.total_time
        }

    def generate_similar_melody(self, original_melody, instrument, similarity=0.7):
        analysis = self.analyze_melody(original_melody)
        new_melody = self.melody_service.compose(instrument, int(analysis['total_duration'] * 4))
        
        adjusted_melody = music_pb2.NoteSequence()
        for note in new_melody.notes:
            new_note = adjusted_melody.notes.add()
            new_note.CopyFrom(note)
            
            pitch_diff = note.pitch - analysis['avg_pitch']
            new_note.pitch = int(analysis['avg_pitch'] + pitch_diff * similarity)
            
            duration = note.end_time - note.start_time
            new_duration = analysis['avg_duration'] + (duration - analysis['avg_duration']) * similarity
            new_note.end_time = new_note.start_time + new_duration

        return adjusted_melody

    def combine_melodies(self, melodies, instruments):
        combined_sequence = music_pb2.NoteSequence()
        
        for i, (melody, instrument) in enumerate(zip(melodies, instruments)):
            for note in melody.notes:
                new_note = combined_sequence.notes.add()
                new_note.MergeFrom(note)
                new_note.instrument = i
                new_note.program = i

        return combined_sequence

    def combine_melodies_with_audio(self, original_melody, new_instrument, similarity=0.7):
        self.logger.info(f"원본 멜로디에 {new_instrument} 멜로디 결합 중...")
        
        new_melody = self.generate_similar_melody(original_melody, new_instrument, similarity)
        combined_sequence = self.combine_melodies([original_melody, new_melody], ['piano', new_instrument])
        midi_path = self.melody_service.create_midi(combined_sequence)
        
        original_audio = self.audio_service.midi_to_wav(midi_path, 'piano')
        new_audio = self.audio_service.midi_to_wav(midi_path, new_instrument)
        
        new_audio = new_audio - 3  # 3dB 낮춤
        combined_audio = original_audio.overlay(new_audio)
        
        output_path = os.path.splitext(midi_path)[0] + f'_with_{new_instrument}.mp3'
        combined_audio.export(output_path, format="mp3")
        
        self.logger.info(f"결합된 멜로디 생성 완료: {output_path}")
        return output_path

    def compose_multi_instrument(self, styles, instruments, num_steps=256, temperature=1.0):
        melodies = [self.melody_service.compose(style, num_steps, temperature) for style in styles]
        combined_sequence = self.combine_melodies(melodies, instruments)
        midi_path = self.melody_service.create_midi(combined_sequence)
        
        mp3_paths = []
        for instrument in instruments:
            mp3_path = self.audio_service.midi_to_mp3(midi_path, instrument=instrument)
            mp3_paths.append(mp3_path)
        
        return mp3_paths

    def compose_with_additional_instrument(self, base_style='piano', additional_instrument='clarinet', num_steps=256, temperature=1.0, similarity=0.7):
        self.logger.info(f"{base_style} 멜로디에 {additional_instrument} 추가 작곡 시작...")
        
        base_melody = self.melody_service.compose(base_style, num_steps, temperature)
        additional_melody = self.melody_service.compose(additional_instrument, num_steps, temperature)
        
        combined_sequence = self.combine_melodies([base_melody, additional_melody], [base_style, additional_instrument])
        midi_path = self.melody_service.create_midi(combined_sequence)
        
        base_audio = self.audio_service.midi_to_wav(midi_path, base_style)
        additional_audio = self.audio_service.midi_to_wav(midi_path, additional_instrument)
        
        # 볼륨 조정 및 오디오 합성
        base_audio = base_audio - 3
        additional_audio = additional_audio - 5
        combined_audio = base_audio.overlay(additional_audio)
        
        # 노이즈 감소 적용
        enhanced_audio = self.enhancement_service.enhance_audio(combined_audio)
        
        # 테스트용 출력 경로 설정
        test_output_path = os.path.join(os.path.dirname(__file__), '..', 'test_outputs')
        os.makedirs(test_output_path, exist_ok=True)
        final_output_path = os.path.join(test_output_path, f"test_{base_style}_with_{additional_instrument}.mp3")
        
        # 결과 저장
        enhanced_audio.export(final_output_path, format="mp3")
        
        self.logger.info(f"작곡 완료: {final_output_path}")
        return final_output_path
    
    def apply_mix_settings(self, midi_data, mix_settings):
        """MIDI 데이터에 믹싱 설정을 적용합니다.

        Args:
            midi_data (bytes): 믹싱할 MIDI 데이터
            mix_settings (dict): 믹싱 설정 (예: volume, pan, EQ, compressor, reverb 등)

        Returns:
            bytes: 믹싱된 오디오 데이터 (MP3 형식)
        """
        self.logger.info("믹싱 설정 적용 중...")

        # MIDI 데이터를 오디오로 변환
        audio = self.midi_to_wav(midi_data)  # AudioSegment 객체

        # 믹싱 설정 적용
        for effect, params in mix_settings.items():
            if effect == 'volume':
                audio = audio + params  # 볼륨 조절 (dB)
            elif effect == 'pan':
                audio = audio.pan(params)  # 팬 설정 (-1.0 ~ 1.0)
            # TODO: EQ, compressor, reverb 등 다른 효과 추가

        # 믹싱된 오디오 데이터를 MP3 형식으로 변환
        mixed_audio_data = audio.export(format="mp3").read()

        self.logger.info("믹싱 완료")
        return mixed_audio_data
    
    def edit_composition(self, composition, edits):
        """NoteSequence 객체를 직접 편집하여 곡을 수정합니다.

        Args:
            composition (NoteSequence): 편집할 음악 데이터 (NoteSequence)
            edits (dict): 편집 내용 (예: {"add_track": {"instrument": "acoustic_grand_piano", "notes": [...]}, "remove_track": 2, ...})

        Returns:
            NoteSequence: 편집된 음악 데이터
        """
        self.logger.info("곡 편집 중...")

        # 트랙 추가
        if "add_track" in edits:
            new_track = music_pb2.NoteSequence()
            new_track.ticks_per_quarter = composition.ticks_per_quarter
            for note_data in edits["add_track"]["notes"]:
                note = new_track.notes.add()
                note.pitch = note_data["pitch"]
                note.start_time = note_data["start_time"]
                note.end_time = note_data["end_time"]
                note.velocity = note_data.get("velocity", 80)  # 기본 velocity 설정
                note.program = sequences_lib.instrument_name_to_program(edits["add_track"]["instrument"])
            composition = sequences_lib.concatenate_sequences([composition, new_track])

        # 트랙 삭제
        if "remove_track" in edits:
            track_index = edits["remove_track"]
            del composition.notes[track_index * composition.notes_per_instrument : (track_index + 1) * composition.notes_per_instrument]

        # TODO: 다른 편집 기능 추가 (음표 추가/삭제, 벨로시티 변경, 악기 변경 등)

        self.logger.info("곡 편집 완료")
        return composition
