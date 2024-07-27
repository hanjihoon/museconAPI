from services.melody_generation_service import MelodyGenerationService
from models.composition_request import CompositionRequest
from utils.music_utils import apply_instrument, adjust_tempo

class CompositionService:
    def init(self):
        self.melody_service = MelodyGenerationService('path/to/model.mag')

    def compose(self, request: CompositionRequest):
        # 장르에 따른 temperature 설정
        temperature = self.get_temperature_for_genre(request.genre)

        # 멜로디 생성
        melody = self.melody_service.generate_melody(num_steps=256, temperature=temperature)

        # 분위기에 따른 조정
        melody = self.adjust_mood(melody, request.mood)

        # 템포 조정
        melody = adjust_tempo(melody, request.tempo)

        # 악기 적용
        for instrument in request.instruments:
            melody = apply_instrument(melody, instrument)

        # MIDI 파일 저장
        outputpath = f"output{request.genre}_{request.mood}.mid"
        self.melody_service.save_composition(melody, output_path)

        return output_path

    def get_temperature_for_genre(self, genre):
        # 장르별 적절한 temperature 반환
        genre_temperature = {
            "pop": 0.8,
            "classical": 0.7,
            "jazz": 0.9,
            "electronic": 1.0
        }
        return genre_temperature.get(genre, 0.8)

    def adjust_mood(self, melody, mood):
        # 분위기에 따라 멜로디 조정 (실제 구현 필요)
        return melody