%% Main script begins here


clear
close all

%% User Parameters
% Assumes 16 bit words, crc is last word
% Gets data from beginning of packet with start_time, till last packet
% start_time + duration
% File time is adjusted if it falls outside of the file boundaries
% Need too read at least 2 packets (duration > 0.01636 s)

load_from_mat_file = 0; %load save frames from mat file
do_save = 0 % save frames
duration = 0.2 %in seconds
shift_start_time = 0; % seconds to shift start time
irig_channel  = 1;
pcm_channel = 6;
derandomize = 1; % derandomization enable
do_crc = 1; % crc error detection enable
remove_errors = 1; % remove detected errors

sync = hex2dec(['FE', '6B', '28', '40'])

frame_len = 48

dirc{1} = 'C:\Users\Administrator\Documents\Work\Customer\RITA\ICRS\'; %need '\' at end
filenamec{1} = 'MTS-2_FC_04262018_TM_ROUND_2_trim.ch10'; %good data!!!!!!

start_timec{1}='116:18:03:04.268'; % 'Day:Hour:Minutes:Seconds'
zero_timec{1}= '116:18:03:04.318';

for filei=1:length(dirc)
    
    % Parameters
    tic;
    filename = filenamec{filei};
    fprintf('\n%s\n', filename);
    zero_time = zero_timec{filei};
    dir = dirc{filei};

    start_seconds = irig2sfbod(start_time) + shift_start_time; %seconds of beginning of data (ignores days)
    zero_seconds = irig2sfbod(zero_time) %seconds to consider as zero for plotting (ignores days)

    if ~load_from_mat_file:
        %% Program constants, computed parameters
        frame_bytes = frame_len * 2;
        rd_len = 50000;
        header_size = 28; % bytes at the beginning of packet before data starts (seemed like it should start 4 bytes earlier)

        full_filename = [dir filename]
        
        %% Relate IRIG time to relative time counter
        fileID = fopen(full_filename)
        
        [fpos, rtime_irig, packet_len] = find_first_sync(rd_len, irig_channel, fileID) % find first irig sync

        time = irig_packet2time(get_packets(1, fpos, irig_channel, packet_len, header_size, 1, fileID, 1, 'uint8', 0)); % get time
        seconds_fbod = time.h * 3600 + time.m * 60 + time.s % seocnds from beginning first time packet
        
        %% Find start packet and npackets
        [ fpos, rtime, packet_len ] = find_first_sync(rd_len, pcm_channel, fileID);

        %find first pcm sync
        [ ~, rtimea ] = get_packets_pcm(1, fpos, pcm_channel, packet_len, header_size, 2, fileID, 1, 0)
        packet_duration = diff(rtimea)/10e6
        irig_time_pcm = seconds_fbod  + (rtime - rtime_irig)/10e6; % irig time of beginning of pcm data
        npackets = ceil(duration/packet_duration)
        nstart = floor((start_seconds - irig_time_pcm)/packet_duration) + 1;
        
        %% estimate the total timeand npacker in file
        [~, rtime_end] = find_last_sync(rd_len, pcm_channel, fileID); % find last packet

        total_time = (rtime_end - rtime) / 10e6
        npackets_max = floor(total_time / packet_duration)
        fprintf('Total file time = %f, npackets_max = %d\n', total_time, npackets_max)
        if nstart < 1
            nstart = 1
            fprintf('Start time delayed to %s!!!\n',sfbod2irig(irig_time_pcm));
        end
        if (nstart + npackets - 1) > npackets_max
            npackets = npackets_max - nstart + 1;
            fprintf('Duration clipped to %f s!!!\n',npackets*packet_duration);
        end

        %% get packets
        [a, rtimea] = get_packets_pcm(1, fpos, pcm_channel, packet_len, header_size, npackets, fileID, nstart, 1);
        % get firs 2 packets
        packets_dt = (rtimea(end) - rtimea(1)) / (npackets - 1) / 10e6;
        byte_dt = packet_dt/(packet_len - header_size);
        
        irig_time_data_start = seconds_fbod + (rtimea(1) - rtime_irig)/10e6; %irig time of beginning of pcm data

        %% Derandomize
        fprintf('Derandomizing....')
        a = typecast(swapbytes(a) 'uint8')
        if derandomize
            a = derandomizer(a)
        end
        