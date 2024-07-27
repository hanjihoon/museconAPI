import os
import json
from dotenv import load_dotenv

def load_config(config_name):
    """
    설정 파일을 로드합니다.
    
    Args:
        config_name (str): 로드할 설정 파일의 이름

    Returns:
        dict: 로드된 설정
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', f'{config_name}.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def load_env_variables():
    """
    .env 파일에서 환경 변수를 로드합니다.
    """
    load_dotenv()

def get_env_variable(var_name, default=None):
    """
    환경 변수를 가져옵니다.

    Args:
        var_name (str): 환경 변수 이름
        default: 환경 변수가 없을 경우 반환할 기본값

    Returns:
        str: 환경 변수 값 또는 기본값
    """
    return os.getenv(var_name, default)