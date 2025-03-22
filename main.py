from chapter10.pcm import PCMF1
from chapter10.computer import ComputerF1
from names import PJson
from chapter10 import C10

from chapter10.util import BitFormat

from tmats import GeneralData, Recorder, PCMFormat

from io import BytesIO
import numpy as np
import datetime

import os
import binascii
import argparse
import json
import re


"""
C10.Packet get_raw body skips over some bytes but buffer has already accounted for that
"""


def explore_tmats(saved_path: str):
    os.system("clear")
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
    os.system("clear")

    fs1 = np.frombuffer(b'\xfe\x6b', dtype=np.uint8)
    fs2 = np.frombuffer(b'\x28\x40', dtype=np.uint8)

    with open(save_path, 'wb') as file:
        data_json = {PJson.time.value: []}
        stream = np.array([], dtype=np.uint8)
        timestamps: list[datetime.datetime] = []
        index = 0
        for packet in C10(ch10_path):

            if packet.data_type == 0x01:
                packet: ComputerF1 = packet
                data_json[PJson.general_data.value] = dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["G"].items()))
                data_json[PJson.recorder.value] = dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["R"].items()))
                data_json[PJson.pcm_format.value] = dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["P"].items()))

            if packet.data_type == 0x09 and index == 123:
                packet: PCMF1 = packet
                raw_body = packet.buffer
                stream = np.concatenate([stream, np.frombuffer(raw_body.read(), dtype=np.uint8)])
                timestamps.append(packet.get_time())
                break
            index += 1

        # stream = np.reshape(stream, (1, stream.shape[0]))
        # np.savetxt("output.txt", stream, fmt="%d")

        extract_frames(stream, fs1, 50, timestamps)

        # file.write(stream)

        # json.dump(data_json, file, indent=4)

def find_sfp_indices(main_list: np.ndarray[np.uint8], sub_list: np.ndarray[np.uint8]):
    shape = (main_list.size - sub_list.size + 1, sub_list.size)
    strides = (main_list.strides[0], main_list.strides[0])
    windows = np.lib.stride_tricks.as_strided(main_list, shape=shape, strides=strides)
    matches = np.all(windows == sub_list, axis=1)
    return np.where(matches)[0]

def extract_frames(stream: np.ndarray[np.uint8],  
                  frame_sync: np.ndarray[np.uint8],
                  frame_bytes: int,
                  timestamps: list[datetime.datetime] = None,
                  bytesdt: int = None):
    
    width = 8
    
    with open("popular.txt", "w") as file:
        for i in range(width):
            print("Bit Shift: {}".format(i))
            left_shift =  np.left_shift(stream, i)
            right_shift = np.right_shift(stream, width - i)
            aligned = np.bitwise_or(left_shift, right_shift)

            words = {}
            for j in range(0, aligned.size//4 * aligned.size, 4):
                hexi = "".join([str(hex(val)) for val in aligned[j:j + 4]])
                if not hexi in words:
                    words[hexi] = 0
                words[hexi] += 1

            ok = max(words, key=words.get)

            print(words)

            file.write("Bitshift: {}, Words: {}, Count: {} \n".format(i, ok, words[ok]))

        # indices = find_sfp_indices(aligned, frame_sync)
        # print("Indices: {}".format(indices))
        # for index in indices:
        #     if index < len(aligned) - frame_bytes - 1:
        #         frame = aligned[index:index + frame_bytes - 1]

def parse_json(json_path: str):
    os.system("clear")
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

    parser.add_argument('-e', '--extract', default="20250307_T4_SIM_LAUNCH_run2.ch10")
    parser.add_argument("-s", '--save')
    parser.add_argument('-p', '--parse')
    parser.add_argument('-t', '--tmats')
    parser.add_argument('-b', '--body')

    args = parser.parse_args()

    if args.extract and args.save:
        extract_ch10(save_path=args.save, ch10_path=args.extract)
    elif args.parse:
        parse_json(args.parse)
    elif args.tmats:
        explore_tmats(args.tmats)
    elif args.body:
        explore_body_pcm(args.body)
    else:
        parser.error("Missing input! Please input either --extract --save or --parse.")