from chapter10.pcm import PCMF1
from chapter10.computer import ComputerF1
from names import PJson
from chapter10 import C10

from tmats import GeneralData, Recorder, PCMFormat

from io import BytesIO
import os
import binascii
import argparse
import json
import re



def explore_tmats(saved_path: str):

    with open(saved_path, "r") as file:
        data = json.load(file)
        
        general_data = GeneralData(data[PJson.general_data.value])
        recorder = Recorder(data[PJson.recorder.value])
        pcm_format = PCMFormat(data[PJson.pcm_format.value])

        while True:
            print("Keyboard Shorcuts: G - General Data, R - Recorder, P - PCM Format, Q - Quit")
            user_input = input("Enter the TMATS section you want to explore: ").strip().lower()
            if user_input == "q":
                os.system("clear")
                break
            if user_input == "g":
                os.system("clear")
                print(repr(general_data))
            if user_input == "r":
                os.system("clear")
                print(repr(recorder))
            if user_input == "p":
                os.system("clear")
                print(repr(pcm_format))
                while True:
                    methods = [name for name in dir(pcm_format) if callable(getattr(pcm_format, name)) and not name.startswith("__")]
                    print("Calls and shortcuts: " + ", ".join(methods) + ", E- Exit")
                    detail_input = input("Enter the PCM Format section you want to explore: ").strip().lower()
                    os.system("clear")
                    if detail_input == "e":
                        break
                    if detail_input in methods:
                        print(f"{detail_input}: " + getattr(pcm_format, detail_input)())



def extract_ch10(save_path: str, ch10_path: str):
    with open(save_path, 'w') as file:
        data_json = {PJson.time.value: []}
        stream = ""
        for packet in C10(ch10_path):

            if packet.data_type == 0x01:
                packet: ComputerF1 = packet
                data_json[PJson.general_data.value] = dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["G"].items()))
                data_json[PJson.recorder.value] = dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["R"].items()))
                data_json[PJson.pcm_format.value] = dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["P"].items()))

            if packet.data_type == 0x09:
                packet: PCMF1 = packet
                raw_body = packet._raw_body()
                buffer = BytesIO(raw_body)
                buffer.seek(36 if packet.secondary_header else 24)
                stream += binascii.hexlify(buffer.read()).decode("ascii")
                data_json[PJson.time.value].append(packet.get_time().isoformat())

        data_json[PJson.stream.value] = stream

        json.dump(data_json, file, indent=4)

def parse_json(json_path: str):
    with open(json_path, "r") as file:
        data = json.load(file)
        raw_body = data[PJson.stream.value]
        buffer = BytesIO(binascii.unhexlify(raw_body))
        
        freq = {}
        while segment := buffer.read(4):
            if not segment in freq:
                freq[segment] = 0
            freq[segment] += 1
        words = max(freq, key=freq.get)
        
        print(f"Words: {binascii.hexlify(words).decode("ascii")}, Count: {freq[words]}")

        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="Decoding Ch10 Files",
        description="Extract contents of Chapter 10 files.",
        epilog="Testing"
    )

    parser.add_argument('-e', '--extract', default="example.ch10")
    parser.add_argument("-s", '--save')
    parser.add_argument('-p', '--parse')
    parser.add_argument('-t', '--tmats')

    args = parser.parse_args()

    if args.extract and args.save:
        extract_ch10(save_path=args.save, ch10_path=args.extract)
    elif args.parse:
        parse_json(args.parse)
    elif args.tmats:
        explore_tmats(args.tmats)
    else:
        parser.error("Missing input! Please input either --extract --save or --parse.")