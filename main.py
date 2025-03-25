from chapter10 import C10
from chapter10.pcm import PCMF1
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
    with open(save_path, 'wb') as file:
        data_json = {}
        for packet in tqdm(C10(ch10_path)):
            if packet.data_type == 0x01:
                packet: ComputerF1 = packet
                data_json[PJson.general_data.value] = GeneralData.parse(packet=packet)
                data_json[PJson.recorder.value] = Recorder.parse(packet=packet)
                data_json[PJson.pcm_format.value] = PCMFormat.parse(packet=packet)
        json.dump(data_json, file, indent=4)

class TMA:
    def __init__(self):
        self.time_count = {}
        self.frame_count = {}

    def incr_tm_cnt(self, channel_id: int):
        if channel_id not in self.time_count.keys():
            self.time_count[channel_id] = 0
        self.time_count[channel_id] += 1
    
    def incr_frm_cnt(self, channel_id: int, amount: int):
        if channel_id not in self.frame_count.keys():
            self.frame_count[channel_id] = 0
        self.frame_count[channel_id] += amount

def get_tmats(ch10_path: str) -> FrameInfo:
    for packet in C10(ch10_path):
        if packet.data_type == 0x01:
            packet: ComputerF1 = packet
            pcm_format = PCMFormat(PCMFormat.parse(packet=packet))
            return FrameInfo(minor_frame_length = pcm_format.P1F1,
                             minor_frame_per_major_frame = pcm_format.P1MFN,
                             frame_sync_pattern = pcm_format.P1MF5)

def extract_memory_allocation(ch10_path: str, info: FrameInfo) -> TMA:
    memory_allocation = TMA()
    for packet in tqdm(C10(ch10_path)):
        memory_allocation.incr_tm_cnt(packet.channel_id)
        if packet.data_type == 0x09:
            packet: PCMF1 = packet
            memory_allocation.incr_frm_cnt(packet.channel_id, BodyStream.get_frame_count(packet.buffer.read(), info))
    return memory_allocation  
            

def extract_ch10(ch10_path: str,
                 verbose: bool,
                 channel: Optional[int],
                 iterations: Optional[int]):
    
    info: FrameInfo = get_tmats(ch10_path)
    memory_alloc = extract_memory_allocation(ch10_path, info)
    times = {channel_id: TimeInfo(channel_id=channel_id, memory_size=size) for channel_id, size in memory_alloc.time_count.items()}

    init_time = None
    prev_time = None
    payloads = {}
    index = 0
    for packet in C10(ch10_path):
        os.system("clear")
        print(index)
        if iterations and iterations == index:
            break
        
        # Assuming tmats is the first packet
        if packet.data_type == 0x01:
            packet: ComputerF1 = packet
            init_time = packet.get_time()
            continue

        # Calculate delta times
        timeinfo = times[packet.channel_id]
        if prev_time != None:
            prev_time.append_delta_time(packet.get_time(), init_time)
        timeinfo.append_time(packet.get_time(), init_time)
        prev_time = timeinfo

        # Decode PCM body if valid
        if not timeinfo.is_end() and packet.data_type == 0x09:
            packet: PCMF1 = packet
            if not packet.channel_id in payloads.keys():
                payloads[packet.channel_id] = BodyStream(times[packet.channel_id], info, memory_alloc.frame_count[packet.channel_id])
            bodystream: BodyStream = payloads[packet.channel_id]
            bodystream.append_body(packet.buffer.read())
        
        index += 1

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

    if args.extract and not args.tmats:
        extract_ch10(ch10_path= args.extract, 
                     verbose= args.verbose, 
                     channel= int(args.channel_id) if args.channel_id else None,
                     iterations = int(args.iterations) if args.iterations else None)
    elif args.tmats:
        explore_tmats(args.tmats)
    else:
        parser.error("Missing input! Please input either --extract --save or --parse.")