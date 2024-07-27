import os

def ensure_directory_exists(directory):
    """
    디렉토리가 존재하지 않으면 생성합니다.

    Args:
        directory (str): 생성할 디렉토리 경로
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_file_extension(filename):
    """
    파일의 확장자를 반환합니다.

    Args:
        filename (str): 파일 이름

    Returns:
        str: 파일 확장자
    """
    return os.path.splitext(filename)[1]

def list_files_in_directory(directory, extension=None):
    """
    지정된 디렉토리의 파일 목록을 반환합니다.

    Args:
        directory (str): 검색할 디렉토리 경로
        extension (str, optional): 특정 확장자로 필터링

    Returns:
        list: 파일 이름 목록
    """
    files = os.listdir(directory)
    if extension:
        return [f for f in files if f.endswith(extension)]
    return files