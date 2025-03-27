def time_drift_visualization(ch10_path: str):
    
    pcm_times = np.array([])
    timef1s = np.array([])
    f1_indices = []
    pcm_indices = []
    index = 0
    offset = None

    for packet in C10(ch10_path):

        if packet.data_type == 0x11:
            packet: TimeF1 = packet
            print(packet.get_time())
            if not offset:
                offset = packet.get_time()
            timef1s = np.append(timef1s, (packet.get_time() - offset).total_seconds())
            f1_indices.append(index)

        if packet.data_type == 0x09:
            print("PCM TIME: {}".format(packet.get_time()))
            packet = PCMF1 = packet
            pcm_times = np.append(pcm_times, (packet.get_time() - offset).total_seconds())
            pcm_indices.append(index)

        index += 1
    
    # Visualize time drift
    # plt.scatter(pcm_indices, pcm_times, color='blue', label='PCM')
    # plt.scatter(f1_indices, timef1s, color='red', label="F1")
    # plt.title("PCM and TimeF1 Timestamps")
    # plt.xlabel("Stream Indices")
    # plt.ylabel("Relative times")

    # Interpolate
    corrected_function = interp1d(f1_indices, timef1s, kind='linear', fill_value='extrapolate')
    interpolated = corrected_function(pcm_indices)
    differences = interpolated - pcm_times
    # plt.scatter(pcm_indices, interpolated, color='green', label='PCM (Interpolated)')
    # plt.show()

    plt.scatter(pcm_times, differences, color='red', label='Residuals')
    plt.title("Time Residuals")
    plt.xlabel("Stream Indices")
    plt.ylabel("Residuals")
    plt.show()
