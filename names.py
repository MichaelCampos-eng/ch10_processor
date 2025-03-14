from enum import Enum

class PJson(Enum):
    time = "timestamps"
    stream = "raw_body"
    general_data = "general_data"
    recorder = "recorder"
    pcm_format = "pcm_format"