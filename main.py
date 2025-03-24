from chapter10 import C10
from chapter10.pcm import PCMF1
from chapter10.computer import ComputerF1

from tmats import GeneralData, Recorder, PCMFormat
from stream import BodyStream, FrameInfo
from names import PJson

from tqdm import tqdm

from typing import Optional
import argparse
import json
import re
import os

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

def extract_ch10(ch10_path: str,
                 save_path: str,
                 verbose: bool,
                 channel: Optional[int],
                 iterations: Optional[int]):
    
    os.system("clear")
    with open(save_path, 'wb') as file:
        data_json = {}
        streams = {}
        index = 0
        for packet in tqdm(C10(ch10_path)):

            if packet.data_type == 0x01:
                packet: ComputerF1 = packet
                data_json[PJson.general_data.value] = dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["G"].items()))
                data_json[PJson.recorder.value] = dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["R"].items()))
                data_json[PJson.pcm_format.value] = dict(map(lambda x: (re.sub(r"[\/\\-]", '', x[0].decode("utf-8")), x[1].decode("utf-8")), packet["P"].items()))

            if packet.data_type == 0x09:
                if iterations and int(iterations) < index:
                    break

                packet: PCMF1 = packet
                if not packet.channel_id in streams.keys():
                    print("\nFound channel id: {}".format(packet.channel_id))
                    streams[packet.channel_id] = BodyStream(packet.channel_id)
                bodystream: BodyStream = streams[packet.channel_id]
                bodystream.append_body(packet.buffer.read())
                bodystream.append_time(packet.get_time())
                BodyStream.save_stream(bodystream.__data__)
                index += 1

        if verbose and channel and channel in streams.keys():
            bodystream: BodyStream = streams[channel]


        # Testing
        print("\nBegin create frames for channel {}\n".format(list(streams.values())[0].channel_id))
        fi = FrameInfo(minor_frame_length=25,
                       bit_per_byte=8,
                       minor_frame_per_major_frame=64)
        list(streams.values())[0].create_frames(fi)

        # json.dump(data_json, file, indent=4)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="Decoding Ch10 Files",
        description="Extract contents of Chapter 10 files.",
        epilog="Testing"
    )

    parser.add_argument('-e', '--extract', default="2025_01_20_ARAV T3_DECOM_4_2_10.ch10")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-c', '--channel_id')
    parser.add_argument('-i', '--iterations')
    parser.add_argument("-s", '--save')
    parser.add_argument('-t', '--tmats')

    args = parser.parse_args()

    if args.extract and args.save:
        extract_ch10(save_path=args.save, 
                     ch10_path=args.extract, 
                     verbose=args.verbose, 
                     channel=args.channel_id,
                     iterations = args.iterations)
    elif args.tmats:
        explore_tmats(args.tmats)
    else:
        parser.error("Missing input! Please input either --extract --save or --parse.")