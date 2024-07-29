# 미사용 기능

이 문서는 현재 프로젝트에서 사용되지 않지만, 향후 사용 가능성이 있는 기능들을 기록합니다.

## 오디오 클리닝 서비스

파일 위치: `backend/services/audio_cleaning_service.py`

이 서비스는 WAV 파일의 노이즈를 제거하고 음질을 개선하는 기능을 제공합니다. 주요 기능은 다음과 같습니다:

- 노이즈 게이트 적용
- 로우패스 필터 적용
- 웨이블릿 디노이징
- 원본 오디오와 처리된 오디오 비교

사용 예시:
```python
from backend.services.audio_cleaning_service import AudioCleaningService

cleaner = AudioCleaningService()
cleaner.clean_audio('input.wav', 'output_cleaned.wav')
cleaner.compare_audio('input.wav', 'output_cleaned.wav', 'comparison.wav')