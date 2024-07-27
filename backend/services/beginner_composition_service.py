from services.melody_generation_service import MelodyGenerationService
from models.composition_request import BeginnerCompositionRequest
from utils.music_utils import apply_instrument, adjust_tempo, adjust_mood

class BeginnerCompositionService:
    def init(self):
        self.melody_service = MelodyGenerationService('path/to/beginner_model.mag')

    def compose(self, request: BeginnerCompositionRequest):
        """
        초보자를 위한 작곡 서비스

        Args:
            request (BeginnerCompositionRequest): 초보자용 작곡 요청 객체

        Returns:
            str: 생성된 MIDI 파일의 경로
        """
        # 장르에 따른 temperature 설정
        temperature = self.get_temperature_for_genre(request.genre)

        # 멜로디 생성 (초보자용 짧은 멜로디)
        melody = self.melody_service.generate_melody(num_steps=128, temperature=temperature)

        # 분위기 조정 (단순화된 버전)
        melody = adjust_mood(melody, request.mood, level='beginner')

        # 템포 조정 (단순화된 옵션)
        melody = adjust_tempo(melody, request.tempo)

        # 악기 적용 (1-2개의 제한된 악기)
        for instrument in request.instruments[:2]:  # 최대 2개 악기만 사용
            melody = apply_instrument(melody, instrument, level='beginner')

        # MIDI 파일 저장
        output_path = f"outputbeginner{request.genre}_{request.mood}.mid"
        self.melody_service.save_composition(melody, output_path)

        return output_path

    def get_temperature_for_genre(self, genre):
        """
        장르별 적절한 temperature 반환 (초보자용 단순화된 버전)
        """
        genre_temperature = {
            "pop": 0.7,
            "classical": 0.6,
            "jazz": 0.8,
            "electronic": 0.9
        }
        return genre_temperature.get(genre, 0.7)  # 기본값은 0.7로 설정 (안전한 선택)