from chapter10 import C10
from chapter10.pcm import PCMF1
from chapter10.packet import Packet
from chapter10.computer import ComputerF1

from tmats import GeneralData, Recorder, PCMFormat
from stream import BodyStream, FrameInfo, TimeInfo
from names import PJson

from tqdm import tqdm

from typing import Optional
import argparse
import json
import os

"""
C10.Packet get_raw body skips over some bytes but buffer has already accounted for that

Assumptions:
- TMATS is in the first packet
- Throughput mode is on
- Little endian for decoding
- Each channel has only 1 packet type
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

def extract_tmats(ch10_path: str,
                 save_path: str):
    with open(save_path, 'w') as file:
        data_json = {}
        for packet in tqdm(C10(ch10_path)):
            if packet.data_type == 0x01:
                packet: ComputerF1 = packet
                data_json[PJson.general_data.value] = GeneralData.parse(packet=packet)
                data_json[PJson.recorder.value] = Recorder.parse(packet=packet)
                data_json[PJson.pcm_format.value] = PCMFormat.parse(packet=packet)
        json.dump(data_json, file, indent=4)

def get_tmats(ch10_path: str) -> FrameInfo:
    for packet in C10(ch10_path):
        if packet.data_type == 0x01:
            packet: ComputerF1 = packet
            pcm_format = PCMFormat(PCMFormat.parse(packet=packet))
            return FrameInfo(minor_frame_length = pcm_format.P1MF1,
                             minor_frame_per_major_frame = pcm_format.P1MFN,
                             frame_sync_pattern = pcm_format.P1MF5)

class MemorySize:
    def __init__(self, ch10_path: str, info: FrameInfo):
        self.time_count = {}
        self.frame_count = {}
        self.__index__ = 0
        self.__extract__(ch10_path, info)

    def incr_tm_cnt(self, channel_id: int):
        if channel_id not in self.time_count.keys():
            self.time_count[channel_id] = 0
        self.time_count[channel_id] += 1
    
    def incr_frm_cnt(self, channel_id: int, amount: int):
        if channel_id not in self.frame_count.keys():
            self.frame_count[channel_id] = 0
        self.frame_count[channel_id] += amount
    
    def decr_frm_cnt(self, channel_id: int, amount: int):
        if channel_id in self.frame_count.keys():
            self.frame_count[channel_id] -= amount
        
    def __extract__(self, ch10_path: str, info: FrameInfo):
        frame_count = None
        channel_id = None
        for packet in tqdm(C10(ch10_path)):
            channel_id = packet.channel_id
            self.incr_tm_cnt(channel_id)
            if packet.data_type == 0x09:
                packet: PCMF1 = packet
                frame_count = BodyStream.get_frame_count(packet.buffer.read(), info)
                self.incr_frm_cnt(channel_id, frame_count)
        if frame_count and channel_id: # Exclude the last packet
            self.decr_frm_cnt(channel_id, frame_count) 

def decode_ch10(ch10_path: str,
                 verbose: bool,
                 channel: Optional[int],
                 iterations: Optional[int]):

    info: FrameInfo = get_tmats(ch10_path)
    payloads: list[int: BodyStream] = {}
    times: list[int: TimeInfo] = {}
    memory_alloc = MemorySize(ch10_path, info)

    for channel_id in memory_alloc.time_count.keys():
        size = memory_alloc.time_count[channel_id]
        print("Size: {}".format(size))
        stream_size = None
        if channel_id in memory_alloc.frame_count.keys():
            stream_size = memory_alloc.frame_count[channel_id]
        times[channel_id] = TimeInfo(channel_id, size, stream_size=stream_size)

    index = 0
    offset = None
    last_packet: Packet = None
    for packet in C10(ch10_path):
        if iterations and iterations == index:
            if channel:
                print(payloads[packet.channel_id])
            break
        index += 1

        # Assuming tmats is the first packet
        if packet.data_type == 0x01:
            packet: ComputerF1 = packet
            offset = packet.get_time()
        
        if last_packet != None:
            # Compute deltas
            last_timeinfo: TimeInfo = times[last_packet.channel_id]
            curr_timeinfo: TimeInfo = times[packet.channel_id]
            curr_timeinfo.append_time(packet.get_time(), offset)
            last_timeinfo.append_delta_time(packet.get_time(), offset)

            # Decode PCM body, excluding the last packet
            if last_packet.data_type == 0x09:
                last_packet: PCMF1 = last_packet
                # If packet is first in its channel, unable to compute times without dt
                if not last_packet.channel_id in payloads.keys():
                    bodystream: BodyStream = BodyStream(last_timeinfo, info, memory_alloc.frame_count[last_packet.channel_id])
                    bodystream.append_body(last_packet.buffer.read())
                    payloads[last_packet.channel_id] = bodystream
                else:
                    bodystream: BodyStream = payloads[last_packet.channel_id]
                    bodystream.compute_times()
                    bodystream.append_body(last_packet.buffer.read())

        last_packet = packet
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="Decoding Ch10 Files",
        description="Extract contents of Chapter 10 files.",
        epilog="Testing"
    )

    parser.add_argument('-d', '--decode', default="2025_01_20_ARAV T3_DECOM_4_2_10.ch10")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-c', '--channel_id')
    parser.add_argument('-i', '--iterations')
    parser.add_argument("-s", '--save')
    parser.add_argument('-et', '--extract_tmats')
    parser.add_argument('-ep', '--explore_tmats')

    args = parser.parse_args()

    if args.decode and not args.extract_tmats and not args.explore_tmats:
        decode_ch10(ch10_path= args.decode, 
                     verbose= args.verbose, 
                     channel= int(args.channel_id) if args.channel_id else None,
                     iterations = int(args.iterations) if args.iterations else None)
    elif args.extract_tmats and not args.explore_tmats:
        extract_tmats(args.extract_tmats, args.save)
    elif args.explore_tmats:
        explore_tmats(args.explore_tmats)
    else:
        parser.error("Missing input! Please input either --extract --save or --parse.")