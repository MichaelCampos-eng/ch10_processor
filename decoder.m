%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ c ] = derandomizer( a )
    %derandomizer: Derandomize PCM packet data.
    % inputs:
    % a - randomized PCM bytes
    % outputs
    % c - derandomized PCM bytes
    n=length(a);
    ab=mydec2bin(a,0); %used to use dec2bin, look at 1p2
    clear a
    reg=false(1,15);
    output=false(1,n*8);
    for i=1:n*8
        if mod(i,10000*8)==0
            fprintf('%d ',i)
        end
        output(i)=xor(ab(i),xor(reg(14),reg(15))); %calc output
        reg=[ab(i) reg(1:end-1)]; %clock update
    end
    fprintf('\n')
    c=mybin2dec_byte3(output);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [bb ] = mydec2bin( b,verbose)
    %% mydec2bin: converts a decimal to binary representation
    % inputs
    % b - decimal bytes
    % verbose - enable bit index printing
    % outputs
    % bb - binary outputs
    bb=false(1,8*length(b));
    for i=1:8
        if verbose
            fprintf('%d ',i);
        end
        bb(i:8:end)=bitget(b,8-i+1)==1;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [a] = mybin2dec_byte3(b)
    % mybin2dec_byte3: convert binary to decimal representation in bytes
    % inputs
    % b - binary input
    % outputs
    % a - bytes
    a=zeros(1,length(b)/8,'uint8');
    b=uint8(b);
    for i=1:8
        a=a+bitshift(b(i:8:end),8-i);
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ fpos,rtime,packet_len ] = find_first_sync( rd_len, channel ,fileID)
    %find_first_sync: returns information about first packet of given channel
    % Inputs
    % rd_len - size of chunks of data to read at a time to search for first sync, 50,000 seems to work ok
    % channel - channel number of packet you're looking for
    % fileID - ID of fileID
    % Outputs

    % fpos - file position of first packet of given channel
    % rtime - relative time counter of first packet
    % packet_len - length of first packet in bytes
    sync=hex2dec(['25'; 'EB'])'; %little endian
    sync=[sync typecast(uint16(channel),'uint8')];
    synci=[];
    fseek(fileID,0,'bof');
    while isempty(synci)
        bytes=fread(fileID,rd_len,'uint8=>uint8')';
        synci=find((bytes(1:end-3)==sync(1)) & (bytes(2:end-2)==sync(2)) & (bytes(3:end-1)==sync(3)) & (bytes(4:end)==sync(4)));
        fseek(fileID, -3, 'cof'); %go back 3 bytes, in case sync is inbetween reads
    end

    bytes = [bytes fread(fileID,24,'uint8=>uint8')']; %make sure have whole header
    rtime = double(typecast([bytes(synci+16:synci+21) 0 0],'uint64')); %relative time of first packet
    packet_len = double(typecast(bytes(synci+4:synci+7),'uint32'));
    fpos = ftell(fileID)+synci(1)+2-rd_len-24;
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [packet_len,rtime ] = find_last_sync( rd_len, channel,fileID )
    %find_last_sync: returns information about first packet of given channel
    % Inputs
    % rd_len - size of chunks of data to read at a time to search for first sync, 50,000 seems to work ok
    % channel - channel number of packet you're looking for%%
    % fileID - ID of fileID
    % Outputs
    % rtime - relative time counter of first packet
    % packet_len - length of first packet in bytes
    sync=hex2dec(['25'; 'EB'])'; %little endian
    sync=[sync typecast(uint16(channel),'uint8')];
    i=1;
    synci=[];
    while isempty(synci)
        if i==1
            fseek(fileID,-rd_len*i, 'eof');
        else
            fseek(fileID,-rd_len*i+3, 'eof'); %go back 3 bytes, in case sync is inbetween reads
            end
        bytes=fread(fileID,rd_len,'uint8=>uint8')';
        synci=find((bytes(1:end-3)==sync(1)) & (bytes(2:end-2)==sync(2)) & (bytes(3:end-1)==sync(3)) & (bytes(4:end)==sync(4)));i=i+1;
    end

    if i>2
        bytes=[bytes fread(fileID,24,'uint8=>uint8')']; %make sure have whole header
    end
    packet_len=double(typecast(bytes(synci+4:synci+7),'uint32'));
    rtime=double(typecast([bytes(synci+16:synci+21) 0 0],'uint64'));
    end

