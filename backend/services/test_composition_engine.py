import unittest
from unittest.mock import MagicMock
from composition_engine import CompositionEngine
from note_seq.protobuf import music_pb2
import os

class TestCompositionEngine(unittest.TestCase):
    def setUp(self):
        self.melody_service = MagicMock()
        self.audio_service = MagicMock()
        self.engine = CompositionEngine(self.melody_service, self.audio_service)

    def test_analyze_melody(self):
        melody = music_pb2.NoteSequence()
        note = melody.notes.add()
        note.pitch = 60
        note.start_time = 0
        note.end_time = 1

        result = self.engine.analyze_melody(melody)
        self.assertIn('avg_pitch', result)
        self.assertIn('pitch_range', result)
        self.assertIn('avg_duration', result)
        self.assertIn('total_duration', result)

    def test_generate_similar_melody(self):
        original_melody = music_pb2.NoteSequence()
        self.melody_service.compose.return_value = music_pb2.NoteSequence()

        result = self.engine.generate_similar_melody(original_melody, 'piano')
        self.assertIsInstance(result, music_pb2.NoteSequence)
        self.melody_service.compose.assert_called_once()

    def test_combine_melodies(self):
        melody1 = music_pb2.NoteSequence()
        melody2 = music_pb2.NoteSequence()
        melody1.notes.add()
        melody2.notes.add()

        result = self.engine.combine_melodies([melody1, melody2], ['piano', 'violin'])
        self.assertEqual(len(result.notes), 2)

    def test_compose_with_additional_instrument(self):
        # Mock 객체 설정
        mock_combined_mp3 = 'mock_combined.mp3'
        self.melody_service.compose.return_value = music_pb2.NoteSequence()
        self.melody_service.create_midi.return_value = 'test.mid'
        self.audio_service.midi_to_audio.return_value = MagicMock()

        # combine_melodies_with_audio 메서드를 모의(mock)로 대체
        self.engine.combine_melodies_with_audio = MagicMock(return_value=mock_combined_mp3)

        # 테스트용 임시 파일 생성
        with open(mock_combined_mp3, 'w') as f:
            f.write('test')

        try:
            result = self.engine.compose_with_additional_instrument()
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))
            print(f"\n테스트 음악 파일이 생성되었습니다: {result}")
        finally:
            # 테스트용 임시 파일 삭제
            if os.path.exists(mock_combined_mp3):
                os.remove(mock_combined_mp3)
            if result and os.path.exists(result):
                os.remove(result)

        self.assertEqual(self.melody_service.compose.call_count, 1)
        self.melody_service.create_midi.assert_not_called()
        self.audio_service.midi_to_audio.assert_not_called()

if name == 'main':
    unittest.main()