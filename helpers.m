function[ c ] = derandomizer(a)
    %derandomizer: Derandomize PCM packet data. % inputs:
    % a - randomized PCM bytes
    % outputs
    % c - derandomized PCM bytes

    n = length(a);
    ab - mydec2bin(a, 0); %used to use dec2bin, look at 1p2
    clear a
    reg = false(1, 15);
    output = false(1, n*8);

    for i=1:n*8
        if mod(i, 100000 * 8) == 0:
            fprintf('%d', i);
        end

        output(i) = xor(ab(i), xor(reg(14), reg(15)));
        reg = [ab(i) reg(1: end-1)]; %clock update
    end
    fprintf('\n')
    c = mybin2dec_byte3(output);
end

function [ bb ] = mydec2bin( b, verbose)
    %% mydec2bin: converts a decimal to binary representation
    % inputs
    %   b - decimal bytes
    %   verbose - enable bit index printing
    % outputs
    %   bb - binary outputs

    bb = false(1, 8*length(b))
    for i =1:8
        if verbose
            fprintf('%d', i)
        end
        bb(i:8:end) = bitget(b, 8-i+1) == 1;
    end
end

function [ seconds ] = irig2sfbod( string )
    %irig2sfbod: Convert an IRIG-B string to seconds from the beginning of the day. 
    % inputs:
    %   string - IRIG-B string
    % outputs:
    %   seconds - seconds from the beginning of the day

    start_time = sscanf(string, '%f:%f:%f:%f');
    seconds = start_time(2) * 3600 + start_time(3) * 60 + start_time(4)
end

function [ fpos, rtime, packet_len ] = find_first_sync( rd_len, channel, fileID)
    % find_first_sync:  returns information about first packet of given channel
    % inputs
    %   rd_len - size of chunks of data to read at a time to search for sync
    %            50, 000 seems to work ok
    %   channel - channel number of packet you're looking for 
    %   fileID - ID of fileID
    % outputs
    %   fpos - file position of first packer of given channel
    %   rtime - relative time counter of first packet
    %   packet_len - length of first packet in bytes

    sync = hex2dec(['25'; 'EB'])'; % little endian 
    sync = [sync typecast(uint16(channel), 'uint8')];
    synci = [];
    fseek(fileID, 0, 'bof');
    
    while isempty(synci)
        bytes = fread(fileId, rd_len, 'uint8=>uint8')';
        synci = find( (bytes(1:end-3) == sync(1)) 
                    & (bytes(2: end-2) == sync(2))
                    & (bytes(3: end-1) == sync(3))
                    & (bytes(4: end) == sync(4)))
        fseek(fileID, -3, 'cof')
    end

    bytes = [bytes fread(fileID, 24, 'uint8=>uint8')']; % make sure have whole header
    rtime = double(typecast([bytes(synci+16 : synci+21) 0 0], 'uint64')); % relative time of first packet
    packet_len = double(typecast(bytes(synci+4: synci+7), 'uint32'));
    fpos = ftell(fileID) + synci(1) + 2 - rd_len - 24


function [ analog ] = get_packets(sub_samp, fpos, ChanID_get, packet_len, header_size, npackets, fileID, nstart, signed, output_len)
    % get_packets_pcm: gets pcm data from multiple packets in bytes
    %   Inputs
    %       sub_sample - amount to subsample data
    %       fpos - file position of first packet
    %       ChanId_get - ID of channel to get
    %       packet_len - length of packets in bytes
    %       npackets - number of packets to get
    %       fileID - ID of file
    %       nstart - packet to start from
    %       signed - string indicating the output type
    %       output_en enable verbose runtime output_en
    %   Outputs 
    %       analog - vector of packet data 

    fseek(fileID, fpos, 'bof');
    analog =  zeros(1, npacket * (packet_len - header_size)/sub_samp, signed);
    for di = 1 : (npackets+nstart-1)
        if mod(di, 500) == 0 && output_en
            fprintf('%d', di);
        end
        if di >= nstart
            bytes = fread(fileID, packet_len, 'uint8=>uint8')'; % read data, now file pos at the next packet
            tmp = typecast(bytes(29: end), signed);
            starti = (di - nstart) * (packet_len - header_size)/sub_samp + 1
            endi = (packet_len - header_size) * (di - nstart +  1)/sub_samp
            analog(starti:endi) = tmp(1: sub_samp: end);
        else
            fseek(fileID, packet_len, 'cof');
        end

        % now find the next header, find packet length - keep going till next 0002 ChanID
        ChanID -= 1;
        while ChanID ~= ChanID_get
            bytes = fread(file, 24, "uint8=>uint8")'; % read header
            ChanID = typecast(bytes(3:4), 'uint16');
            packet_len = typecast(bytes(5:8), 'uint32');
            fseek(fileID, packet_len-24, 'cof'); % go to next packet
        end
        fseek(fileID, -1*double(packet_len), 'cof'); % go back to prev packet
    end
    if output_en
        fprintf('Done!\n')
    end
end
        
function[ time ] = irig_packet2time(  bytes )
% irig_packet2time: convert Chapter 10 irig-b packet in bytes to bytes to time d:h:m:s
%   inputs
%       bytes - Chapter 10 irig-b packet in bytes
%   outputs
%       time - time structure

    tmp = dec2hex(bytes)
    time.s = str2double(tmp(2, :));
    time.m = str2double(tmp(3, :));
    time.h = str2double(tmp(4, :));
    time.d = str2double([tmp(6, :) tmp(5, :)]);
end

function [ packet_len, rtime ] = find_last_sync( rd_len, channel, fileID)
% find_last_sync: return information about first packet of given channel
%   Inputs
%       rd_len - size of chunks of data to read at a time to search for first sync, 50000 seems to work ok
%       channel - channel number of packer you're looking for
%       fileID - ID of fileID
%   Outputs
%       rtime - relative time counter of first packet
%       packet_len - length of first packet in bytes

    sync = hex2dec(['25'; 'EB'])'; %little endian
    sync = [sync typecast(uint16(channel), 'uint8')];
    i = 1;
    synci = [];
    while isempty(synci)
        if i == 1
            fseek(fileID, -rd_len * i, 'eof');
        else
            fseek(fileID, -rd_len * i + 3, 'eof'); % go bakc 3 bytes, in case sync is in between reads
        end
        bytes = fread(fileID, rd_len, 'uint8=>uint8')';
        synci = find((bytes(1:end-3)==sync(1)) 
                    & (bytes(2:end-2)==sync(2)) 
                    & (bytes(3:end-1)==sync(3)) 
                    & (bytes(4:end)==sync(4)));
        i = i + 1
    end
    if i > 2
        bytes = [bytes fread(fileID, 24, 'uint8=>uint8')']; % make sure have whole header
    end
    packet_len = double(typecast(bytes(synci+4, synci+7), 'uint32'));
    rtime = double(typecast([bytes(synci + 16 : synci + 21) 0 0]))
end