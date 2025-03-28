from chapter10 import C10
from chapter10.pcm import PCMF1
from chapter10.packet import Packet
from chapter10.computer import ComputerF1
from chapter10.time import TimeF1

from tmats import GeneralData, Recorder, PCMFormat
from stream import BodyStream, FrameInfo, TimeInfo, MemorySize
from names import PJson
import numpy as np
import pandas as pd

from tqdm import tqdm

from numpy.lib.stride_tricks import sliding_window_view

from typing import Optional
import argparse
import json
import os
import time
import sys

"""
C10.Packet get_raw body skips over some bytes but buffer has already accounted for that

Assumptions:
- TMATS is in the first packet
- Throughput mode is on
- Little endian for decoding
- Each channel has only 1 packet type

Clarity: 
- TMATS packet time is the current time one is decoding the file stream
- TimeF1 should be used as a time anchor point for PCM packets
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

def view_csv(csv_pth: str):
    df: pd.DataFrame = pd.read_csv(csv_pth)
    uniques = pd.Series(df.values.ravel()).unique()
    counts = pd.Series(df.values.ravel()).value_counts() 
    masks: dict[str: np.ndarray[np.uint]] = {key: (key == df.values).astype(int) for key in uniques}

    stream = np.load("decoded_frames.npy").ravel()
    time = np.load("decoded_times.npy")

    sfid = masks["SFID"]
    
    #memory allocation
    data_count = counts.ilocp["SFID"] * stream.shape[1] // 64
    data_times = np.empty((data_count, ), dtype=np.longdouble)
    data = np.empty( (data_count, ), dtype=np.longdouble)
    
    
    

    print(time)
    

def decode_ch10(ch10_path: str,
                 verbose: bool,
                 channel: Optional[int],
                 iterations: Optional[int]):
    
    
    info: FrameInfo = get_tmats(ch10_path)
    start = time.time()
    memory = MemorySize(ch10_path, info)
    end = time.time()
    print("Time elapsed for memory extraction: {}".format(end - start))

    # Set up times
    times: list[int: TimeInfo] = {}
    for channel_id, time_size in memory.t_size.items():
        stream_size = memory.f_size[channel_id] if channel_id in memory.f_size.keys() else None
        times[channel_id] = TimeInfo(time_size, stream_size=stream_size)
    
    # Set up body streams
    payloads: list[int: BodyStream] = {}
    for channel_id, stream_size in memory.f_size.items():
        payloads[channel_id] = BodyStream(times[channel_id], info, stream_size)

    index = 0
    offset = None
    last_packet: Packet = None
    start = time.time()
    for packet in C10(ch10_path):

        if packet.data_type == 0x01:
            continue

        if iterations and iterations == index:
            end = time.time()
            stream: BodyStream = payloads[channel]
            print("Time Elapsed for decoding: {}".format(end - start))
            print("Channel ID: {}".format(packet.channel_id))
            print(stream)
            np.save("decoded_frames.npy", stream.__frames__)
            np.save("decoded_times.npy", stream.time_info.relative_times)
            break
        index += 1

        # Assuming first TimeF1 packet comes before any PCM packet
        if packet.data_type == 0x11 and not offset:
            packet: TimeF1 = packet
            offset = packet.get_time()
        
        if last_packet != None:
            # Compute deltas
            curr_timeinfo: TimeInfo = times[packet.channel_id]
            last_timeinfo: TimeInfo = times[last_packet.channel_id]
            last_timeinfo.append_delta_time(packet.get_time(), offset)
            curr_timeinfo.append_time(packet.get_time(), offset)
            
            # Compute time for each word
            if last_packet.data_type == 0x09:
                last_packet: PCMF1 = last_packet
                bodystream: BodyStream = payloads[last_packet.channel_id]
                # If packet is first in its channel, unable to compute times without dt
                if not bodystream.is_empty:
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
    parser.add_argument('-csv', '--import_csv')

    args = parser.parse_args()

    if args.import_csv:
        view_csv(args.import_csv)

    elif args.decode and not args.extract_tmats and not args.explore_tmats and not args.import_csv:
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