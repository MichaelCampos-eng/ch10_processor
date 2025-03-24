import binascii

class TMATS:
    
    def __init__(self, data : dict):
        for key, value in data.items():
            self.__setattr__(key, value)
    
    def __repr__(self) -> str:
        rep = "Attributes Overview \n \n"
        for key, val in vars(self).items():
            rep += f"{key}: {val}\n"
        return rep

class GeneralData(TMATS):
    def __init__(self, data: dict):
        super().__init__(data)

class Recorder(TMATS):
    def __init__(self, data: dict):
        super().__init__(data)

class PCMFormat(TMATS):
    def __init__(self, data: dict):
        super().__init__(data)

    def get_minor_frame_sync_offset(self):
        return self.P1MF4

    def get_sync_pattern(self):
        bin = "Binary: {}".format(self.P1MF5)
        hex = "Hex: {}".format(binascii.hexlify(self.P1MF5.encode())).decode()
        return bin + "\n" + hex

    def get_words_per_minor_frame(self):
        return self.P1F1 
    
    def get_minor_frame_count_per_major_frame(self):
        return self.P1MFN
    
    def get_bits_per_minor_frame(self):
        return self.P1MF2
    
    def get_word_length(self):
        return self.P1IDC21
    
    def get_word_count(self):
        return self.P1IDC41
    
    def get_word_position(self):
        return self.P1IDC11