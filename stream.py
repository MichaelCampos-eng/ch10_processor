import numpy as np
import datetime


class FrameInfo:

    def __init__(self, minor_frame_length: int,
                 bit_per_byte: int,
                 minor_frame_per_major_frame: int,
                 frame_sync_pattern: str = '11111110011010110010100001000000'):
        self.minor_frame_length = minor_frame_length
        self.bit_per_byte = bit_per_byte
        self.minor_frame_per_major_frame = minor_frame_per_major_frame
        self.frame_sync = FrameInfo.format_frame_sync(frame_sync_pattern)
    
    @property
    def frame_byte_length(self) -> int:
        return self.minor_frame_length * 2
    
    @property
    def bit_per_word(self) -> int:
        return self.bit_per_word * 2

    @staticmethod
    def format_frame_sync(fs_bin: str) -> np.ndarray[np.uint8]:
        byte_width = 8
        fs_dec = np.array([int(fs_bin[i:i+byte_width], 2) for i in range(0, len(fs_bin), byte_width)], dtype=np.uint8)
        return np.concatenate([fs_dec[:2][::-1], fs_dec[2:][::-1]])

class BodyStream:

    def __init__(self, channel_id, data: bytes = b''):
        self.channel_id = channel_id
        self.__data__ = np.frombuffer(buffer=data, dtype=np.uint8)
        self.__timestamps__: list[datetime.datetime] = []
    
    def append_body(self, new_data: bytes):
        new_data = np.frombuffer(buffer=new_data, dtype=np.uint8)
        self.__data__ = np.concatenate([self.__data__, new_data])

    def append_time(self, timestamp: datetime.datetime):
        self.__timestamps__.append(timestamp)
    
    def __find_sfp_indices__(self, main_list: np.ndarray[np.uint8], sub_list: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
        shape = (main_list.size - sub_list.size + 1, sub_list.size)
        strides = (main_list.strides[0], main_list.strides[0])
        windows = np.lib.stride_tricks.as_strided(main_list, shape=shape, strides=strides)
        matches = np.all(windows == sub_list, axis=1)
        return np.where(matches)[0]
    
    def __align__(self, shift, bits):
        left_shift = np.left_shift(self.__data__, shift)
        right_shift = np.right_shift(self.__data__, bits - shift)
        return np.bitwise_or(left_shift, right_shift)
    
    def __extract_frames__(self, info: FrameInfo):
        potentials_indices = {}
        for i in range(info.bit_per_byte):
            aligned = self.__align__(i, info.bit_per_byte)
            potentials_indices[i] = self.__find_sfp_indices__(aligned, info.frame_sync)

        shift, indices = max(potentials_indices.items(), key=lambda x: len(x[1]))
        indices = indices[indices >= info.frame_byte_length]
        aligned = self.__align__(shift, info.bit_per_byte)
        frames = np.empty((info.minor_frame_length, len(indices)), dtype=np.uint16)
        for count, index in enumerate(indices):
            frames[:, count] = aligned[index - info.frame_byte_length + 4: index + 4].astype(np.uint8).view('<u2')

        print(self.__timestamps__)


    def create_frames(self, info: FrameInfo):
        frames = self.__extract_frames__(info)

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