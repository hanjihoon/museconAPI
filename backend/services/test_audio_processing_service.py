import unittest
from unittest.mock import patch, MagicMock
from audio_processing_service import AudioProcessingService
from pydub import AudioSegment

class TestAudioProcessingService(unittest.TestCase):
    def setUp(self):
        self.service = AudioProcessingService()

    def test_load_soundfonts(self):
        soundfonts = self.service._load_soundfonts()
        self.assertIsInstance(soundfonts, dict)
        self.assertGreater(len(soundfonts), 0)

    def test_get_instrument_program(self):
        result = self.service._get_instrument_program('piano')
        self.assertEqual(result, 'uprightpianokw-20220221')

    @patch('audio_processing_service.subprocess.run')
    @patch('audio_processing_service.AudioSegment.from_wav')
    def test_midi_to_audio(self, mock_from_wav, mock_run):
        mock_run.return_value = MagicMock(stdout="Test output")
        mock_from_wav.return_value = AudioSegment.silent(duration=1000)

        result = self.service.midi_to_audio('test.mid', 'piano')
        self.assertIsInstance(result, AudioSegment)
        mock_run.assert_called_once()
        mock_from_wav.assert_called_once()

    @patch('audio_processing_service.AudioProcessingService.midi_to_audio')
    def test_midi_to_mp3(self, mock_midi_to_audio):
        mock_midi_to_audio.return_value = AudioSegment.silent(duration=1000)
        result = self.service.midi_to_mp3('test.mid', 'piano')
        self.assertTrue(result.endswith('.mp3'))

if name == 'main':
    unittest.main()