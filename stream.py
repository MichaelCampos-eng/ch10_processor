import numpy as np
import datetime
import matplotlib.pyplot as plt
from chapter10 import C10
from typing import Optional
from tqdm import tqdm
from chapter10.pcm import PCMF1


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
        self.minor_frame_length = int(minor_frame_length) + 1
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
    
    nanoseconds = 1e9

    def __init__(self, time_size: int, stream_size: Optional[int]):
        self.__t_stamps__: np.ndarray[np.uint64] = np.empty((time_size,), dtype=np.uint64)
        self.__delta_times__: np.ndarray[np.uint64] = np.empty((time_size - 1,), dtype=np.uint64)
        self.relative_times: Optional[np.ndarray[np.uint64]] = np.empty((stream_size, ), np.uint64) if stream_size else None

        self.__ts_index__: int = 0
        self.__dt_index__: int = 0

    def append_time(self, timestamp: datetime.datetime, offset: datetime.datetime):
        if self.__ts_index__ < self.__t_stamps__.size:
            self.__t_stamps__[self.__ts_index__] = (timestamp - offset).total_seconds() * TimeInfo.nanoseconds
            self.__ts_index__ += 1
    
    def append_delta_time(self, new_time: datetime.datetime, offset: datetime.datetime):
        if self.__dt_index__ < self.__delta_times__.size:
            self.__delta_times__[self.__dt_index__] = (new_time - offset).total_seconds() * TimeInfo.nanoseconds - self.get_ts()
            self.__dt_index__ += 1
    
    # Retrieves the latest dt appended
    def get_dt(self):
        return self.__delta_times__[self.__dt_index__ - 1]

    # Retrieves current relative timestamp
    def get_ts(self) -> np.uint16:
        return self.__t_stamps__[self.__ts_index__ - 1]

    @property
    def count(self):
        return self.__delta_times__.size

class BodyStream:

    def __init__(self, time_info: TimeInfo, frame_info: FrameInfo, memory: int):
        self.time_info: TimeInfo = time_info
        self.frame_info: FrameInfo = frame_info
        self.__frames__ = np.empty((frame_info.minor_frame_length, memory), dtype=np.uint16)
        self.__delta__: int = None
        self.__index__ = 0
    
    def __str__(self):
        frames_contents = "Number of frames: {}, Frame length: {}\n".format(self.__frames__.shape[1], self.__frames__.shape[0])
        time_contents = "Number of time deltas: {}\n".format(self.time_info.count)
        dt_contents = "Number of times: {}\n".format(self.time_info.relative_times.shape)
        divider = "#" * 50
        return frames_contents + time_contents  + dt_contents + divider
    
    def append_body(self, new_data: bytes):
        new_data = np.frombuffer(buffer=new_data, dtype=np.uint8)
        frames = BodyStream.extract_frames(new_data, self.frame_info)
        self.__delta__ = frames.shape[1]
        self.__frames__[:,self.__index__: self.__index__ + self.__delta__] = frames
        self.__index__ += self.__delta__

    def compute_times(self):
        sub_times = np.arange(self.__delta__) * self.time_info.get_dt() / self.__delta__ + self.time_info.get_ts()
        self.time_info.relative_times[self.__index__ - self.__delta__ : self.__index__] = sub_times
    
    @property
    def is_empty(self):
        return self.__index__ == 0
    
    @staticmethod
    def extract_frames(data: np.ndarray[np.uint8], info: FrameInfo) -> np.ndarray[np.uint8]:
        potentials_indices = {}
        for i in range(info.byte_width):
            aligned = BodyStream.__align__(data, i, info.byte_width)
            potentials_indices[i] = BodyStream.__find_sfp_indices__(aligned, info.frame_sync)
        shift, indices = max(potentials_indices.items(), key=lambda x: len(x[1]))
        indices = indices[indices >= info.frame_byte_length]
        aligned = BodyStream.__align__(data, shift, info.byte_width)
        frames = np.empty((info.minor_frame_length, len(indices)), dtype=np.uint16)
        for count, index in enumerate(indices):
            frames[:, count] = aligned[index - info.frame_byte_length + 4: index + 4].view('<u2').astype(np.uint16)
        return frames

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
    def get_frame_count(data: bytes, info: FrameInfo):
        return  BodyStream.extract_frames(np.frombuffer(data, np.uint8), info).shape[1]
        
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

class MemorySize:
    def __init__(self, ch10_path: str, info: FrameInfo):
        self.t_size: dict[int: int] = {}
        self.f_size: dict[int: int] = {}
        self.__index__ = 0
        self.__extract__(ch10_path, info)

    def __incr_time__(self, channel_id: int):
        if channel_id not in self.t_size.keys():
            self.t_size[channel_id] = 0
        self.t_size[channel_id] += 1
    
    def __incr_frame__(self, channel_id: int, amount: int):
        if channel_id not in self.f_size.keys():
            self.f_size[channel_id] = 0
        self.f_size[channel_id] += amount
        
    def __extract__(self, ch10_path: str, info: FrameInfo):
        for packet in tqdm(C10(ch10_path)):
            if not packet.data_type == 0x01:
                self.__incr_time__(packet.channel_id)
            if packet.data_type == 0x09:
                packet: PCMF1 = packet
                frame_count = BodyStream.get_frame_count(packet.buffer.read(), info)
                self.__incr_frame__(packet.channel_id, frame_count)