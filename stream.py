import numpy as np
import datetime
import matplotlib.pyplot as plt

'''
Because there is no end time for the last PCM packet, we discard this packet
and only use its start time as end time for the second-to-last packet
'''

class FrameInfo:

    byte_width = 8
    def __init__(self, 
                 minor_frame_length: str,
                 minor_frame_per_major_frame: str,
                 frame_sync_pattern: str = '11111110011010110010100001000000'):
        self.minor_frame_length = int(minor_frame_length)
        self.minor_frame_per_major_frame = int(minor_frame_per_major_frame)
        self.frame_sync = FrameInfo.format_frame_sync(frame_sync_pattern)
    
    @property
    def frame_byte_length(self) -> int:
        return self.minor_frame_length * 2
    
    @property
    def bit_per_word(self) -> int:
        return self.bit_per_word * 2

    @staticmethod
    def format_frame_sync(fs_bin: str) -> np.ndarray[np.uint8]:
        fs_dec = np.array([int(fs_bin[i:i+FrameInfo.byte_width], 2) for i in range(0, len(fs_bin), FrameInfo.byte_width)], dtype=np.uint8)
        return np.concatenate([fs_dec[:2][::-1], fs_dec[2:][::-1]])

class TimeInfo:
    
    def __init__(self, channel_id, memory_size: int):
        self.channel_id = channel_id
        self.__t_stamps__: np.ndarray[np.longdouble] = np.empty((memory_size,), dtype=np.longdouble)
        self.__delta_times__: np.ndarray[np.longdouble] = np.empty((memory_size - 1,), dtype=np.longdouble)

        self.__ts_index__: int = 0
        self.__dt_index__: int = 0

    
    def append_time(self, timestamp: datetime.datetime, offset: datetime.datetime):
        self.__t_stamps__[self.__ts_index__] = (timestamp - offset).total_seconds()
    
    def append_delta_time(self, endtime: datetime.datetime, offset: datetime.datetime):
        diff_secs = (endtime - offset).total_seconds() - self.__t_stamps__[self.__ts_index__]
        self.__delta_times__[self.__dt_index__] = diff_secs
        self.__dt_index__ += 1
        self.__ts_index__ += 1

    def is_end(self):
        return self.__dt_index__ >= self.__delta_times__.size - 1

    @property
    def count(self):
        return len(self.__t_stamps__)
    
    def __next__(self):
        if self.__index__ < self.count:
            timestamp = self.__t_stamps__[self.__ts_index__]
            self.__index__ += 1
            return timestamp
        else:
            StopIteration
    
    def __iter__(self):
        self.__ts_index__ = 0
        return self

class BodyStream:

    def __init__(self, time_info: TimeInfo, frame_info: FrameInfo, memory: int):
        self.__timestamps__: TimeInfo = time_info
        self.__info__: FrameInfo = frame_info
        self.__data__ = np.empty((frame_info.minor_frame_length, memory - 1), dtype=np.uint8)
        self.__index__ = 0
        self.__frames__ = None

    @property
    def channel_id(self):
        return self.__timestamps__.channel_id
    
    def append_body(self, new_data: bytes):
        new_data = np.frombuffer(buffer=new_data, dtype=np.uint8)
        self.__data__[:,self.__index__] = new_data
        self.__index__ += 1

    def count(self):
        return self.__timestamps__.count
    
    def create_frames(self, info: FrameInfo):
        self.__frames__ = BodyStream.extract_frames(info)

    @staticmethod
    def __find_sfp_indices__(main_list: np.ndarray[np.uint8], sub_list: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
        shape = (main_list.size - sub_list.size + 1, sub_list.size)
        strides = (main_list.strides[0], main_list.strides[0])
        windows = np.lib.stride_tricks.as_strided(main_list, shape=shape, strides=strides)
        matches = np.all(windows == sub_list, axis=1)
        return np.where(matches)[0]
    
    @staticmethod
    def __align__(data: np.ndarray[np.uint8], shift: int, bits: int):
        left_shift = np.left_shift(data, shift)
        right_shift = np.right_shift(data, bits - shift)
        return np.bitwise_or(left_shift, right_shift)
    
    @staticmethod
    def __extract_frames__(data: np.ndarray[np.uint8], info: FrameInfo) -> np.ndarray[np.uint16]:
        potentials_indices = {}
        for i in range(info.byte_width):
            aligned = BodyStream.__align__(data, i, info.byte_width)
            potentials_indices[i] = BodyStream.__find_sfp_indices__(aligned, info.frame_sync)
        shift, indices = max(potentials_indices.items(), key=lambda x: len(x[1]))
        indices = indices[indices >= info.frame_byte_length]
        aligned = BodyStream.__align__(data, shift, info.byte_width)
        frames = np.empty((info.minor_frame_length, len(indices)), dtype=np.uint16)
        for count, index in enumerate(indices):
            frames[:, count] = aligned[index - info.frame_byte_length + 4: index + 4].astype(np.uint8).view('<u2')
        return frames
    
    @staticmethod
    def get_frame_count(data: bytes, info: FrameInfo):
        return  BodyStream.__extract_frames__(np.frombuffer(data, np.uint8), info).shape[1]
        
    @staticmethod
    def save_stream(stream: np.ndarray[np.uint8]):
        stream = np.reshape(stream, (1, stream.shape[0]))
        np.savetxt("output.txt", stream, fmt="%d")

    @staticmethod
    def derandomizer(stream: np.ndarray[np.uint8]):
        size = stream.size
        stream_bits = np.unpackbits(stream)
        reg = np.zeros(15, dtype=np.uint8)
        output = np.zeros(size * 8, dtype=np.uint8)
        for i in range(size * 8):
            output[i] = np.bitwise_xor(stream_bits[i], np.bitwise_xor(reg[13], reg[14]))
            reg[1:] = reg[:-1]
            reg[0] = stream_bits[i]
        return np.packbits(output)