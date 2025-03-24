from chapter10.pcm import PCMF1
from chapter10.computer import ComputerF1
from names import PJson
from chapter10 import C10

from tmats import GeneralData, Recorder, PCMFormat

import numpy as np
from tqdm import tqdm

from io import BytesIO
import os
import binascii
import bitstruct
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

def explore_pcm_packet(saved_path: str):
    with open(saved_path, "r") as file:
        data = json.load(file)
        packet = PCMF1(**data[PJson.pcm_packet.value])
        while True:
            print("Keyboard shorcuts: 'Q' - Quit\n\nAttributes: " + ", ".join(vars(packet).keys()))
            user_input = input("Enter: ").strip().lower()
            if user_input == "q":
                os.system("clear")
                break
            elif user_input in vars(packet):
                os.system("clear")
                print(f"{user_input}: " +  str(getattr(packet, user_input)) + "\n")
            else:
                os.system("clear")
                print("Invalid attribute\n")

def explore_body_pcm(saved_path: str):
    with open(saved_path, "r") as file:
        data = json.load(file)
        stream = data[PJson.stream.value]
        for i in range(0, len(stream) - 8, 1):
            segment = stream[i: i+8]
            if "fe6b" in segment:
                print(segment)

def extract_ch10(save_path: str, ch10_path: str):
    first = True
    with open(save_path, 'w') as file:
        data_json = {PJson.time.value: []}
        
        sync_frame_pattern = "11111110011010110010100001000000"
        sfp_np = np.array([np.uint8(bit) for bit in sync_frame_pattern])
        stream = np.array([])
        count = 0
        for packet in tqdm(C10(ch10_path)):

            if packet.data_type == 0x01:
                packet: ComputerF1 = packet
                data_json[PJson.general_data.value] = dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["G"].items()))
                data_json[PJson.recorder.value] = dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["R"].items()))
                data_json[PJson.pcm_format.value] = dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["P"].items()))
            
            """
            Only extracts the raw body of PCM packet excluding CSDW
            """
            if packet.data_type == 0x09:
                packet: PCMF1 = packet
                raw_body = packet._raw_body()

                buffer = BytesIO(raw_body)
                buffer.seek(36 if packet.secondary_header else 24)

                payload: bytes = buffer.read()
                stream = np.concatenate([stream, np.unpackbits(np.frombuffer(payload, dtype=np.uint8))])

                if count == 1000:
                    break
                count += 1

                # stream += binascii.hexlify(payload).decode("ascii")
                # data_json[PJson.time.value].append(packet.get_time().isoformat())

                # if first:
                #     attri = vars(packet)
                #     print(payload)
                #     attri["payload"] = payload
                #     del attri["parent"]
                #     del attri["buffer"]
                #     del attri["Message"]
                #     data_json[PJson.pcm_packet.value] = attri
                #     first = False
        for i in range(0, stream.shape[0] - sfp_np.shape[0], 1):
            if np.all(sfp_np == stream[i:i+sfp_np.shape[0]]):
                print("Starting index: {}".format(i))

        # data_json[PJson.stream.value] = stream
        # json.dump(data_json, file, indent=4)

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

    parser.add_argument('-e', '--extract', default="2025_01_20_ARAV T3_DECOM_4_2_10.ch10")
    parser.add_argument("-s", '--save')
    parser.add_argument('-p', '--parse')
    parser.add_argument('-t', '--tmats')
    parser.add_argument('-b', '--body')
    parser.add_argument('-pp', '--pcm')

    args = parser.parse_args()

    if args.extract and args.save:
        extract_ch10(save_path=args.save, ch10_path=args.extract)
    elif args.parse:
        parse_json(args.parse)
    elif args.tmats:
        explore_tmats(args.tmats)
    elif args.body:
        explore_body_pcm(args.body)
    elif args.pcm:
        explore_pcm_packet(args.pcm)
    else:
        parser.error("Missing input! Please input either --extract --save or --parse.")