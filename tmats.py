from chapter10.computer import ComputerF1
import re

class TMATS:
    
    def __init__(self, data : dict):
        for key, value in data.items():
            self.__setattr__("{}".format(key), value)
    
    def __repr__(self) -> str:
        rep = "Attributes Overview \n \n"
        for key, val in vars(self).items():
            rep += f"{key}: {val}\n"
        return rep

class GeneralData(TMATS):

    def __init__(self, data: dict):
        super().__init__(data)

    @staticmethod
    def parse(packet: ComputerF1):
        return dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["G"].items()))

class Recorder(TMATS):

    def __init__(self, data: dict):
        super().__init__(data)

    @staticmethod
    def parse(packet: ComputerF1):
        return dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["R"].items()))

class PCMFormat(TMATS):
    
    def __init__(self, data: dict):
        super().__init__(data)

    @staticmethod
    def parse(packet: ComputerF1):
        return  dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["P"].items()))