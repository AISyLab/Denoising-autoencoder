import os
import os.path
import sys
import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
import shutil

def load_ascad(ascad_database_file):
    check_file_exists(ascad_database_file)
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." %
              ascad_database_file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
    X_profiling = X_profiling.reshape(
        (X_profiling.shape[0], X_profiling.shape[1], 1))
    # X_metadata = in_file['Profiling_traces/metadata/']
    # X_desync = X_metadata['desync']
    # print(X_desync)

    # Load attack traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
    X_metadata = in_file['Attack_traces/metadata/']
    return (X_profiling, X_attack)

def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

def show(plot):
    plt.plot(plot)
    plt.show(block=True)

def addShuffling(traces, shuffling_level):
    print('Add shuffling...')
    output_traces = np.zeros(np.shape(traces))
    print(np.shape(output_traces))
    for idx in range(len(traces)):
        if(idx % 2000 == 0):
            print(str(idx) + '/' + str(len(traces)))
        trace = traces[idx]
        for iter in range(shuffling_level):
            rand = np.random.randint(low=0, high=len(traces[0]), size=(2,))
            temp = trace[rand[0]]
            trace[rand[0]] = trace[rand[1]]
            trace[rand[1]] = temp
        output_traces[idx] = trace
    return output_traces

def addClockJitter(traces, clock_range, trace_length):
    print('Add clock jitters...')
    output_traces = []
    min_trace_length = 100000
    for trace_idx in range(len(traces)):
        if(trace_idx % 2000 == 0):
            print(str(trace_idx) + '/' + str(len(traces)))
        trace = traces[trace_idx]
        point = 0
        new_trace = []
        while point < len(trace)-1:
            new_trace.append(int(trace[point]))
            # generate a random number
            r = random.randint(-clock_range, clock_range)
            # if r < 0, delete r point afterward
            if r <= 0:
                point += abs(r)
            # if r > 0, add r point afterward
            else:
                avg_point = int((trace[point] + trace[point+1])/2)
                for _ in range(r):
                    new_trace.append(avg_point)
            point += 1
        output_traces.append(new_trace)
    return regulateMatrix(output_traces, trace_length)

def addRandomDelay(traces, a, b, jitter_amplitude, trace_length):
    print('Add random delays...')
    output_traces = []
    min_trace_length = 100000
    for trace_idx in range(len(traces)):
        if(trace_idx % 2000 == 0):
            print(str(trace_idx) + '/' + str(len(traces)))
        trace = traces[trace_idx]
        point = 0
        new_trace = []
        while point < len(trace)-1:
            new_trace.append(int(trace[point]))
            # generate a random number
            r = random.randint(0, 10)
            # 10% probability of adding random delay
            if r > 5:
                m = random.randint(0, a-b)
                num = random.randint(m, m+b)
                if num > 0:
                    for _ in range(num):
                        new_trace.append(int(trace[point]))
                        new_trace.append(int(trace[point]+jitter_amplitude))
                        new_trace.append(int(trace[point+1]))
            point += 1
            # if len(new_trace) > trace_length:
            #    break
        output_traces.append(new_trace)

    return regulateMatrix(output_traces, trace_length)

def addGussianNoise(traces, noise_level):
    print('Add Gussian noise...')
    if noise_level == 0:
        return traces
    else:
        output_traces = np.zeros(np.shape(traces))
        print(np.shape(output_traces))
        for trace in range(len(traces)):
            if(trace % 5000 == 0):
                print(str(trace) + '/' + str(len(traces)))
            profile_trace = traces[trace]
            noise = np.random.normal(
                0, noise_level, size=np.shape(profile_trace))
            output_traces[trace] = profile_trace + noise
        return output_traces

# A function to make sure the noisy traces has same length (padding zeros)
def regulateMatrix(M, size):
    maxlen = max(len(r) for r in M)
    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        if len(row) <= maxlen:
            Z[enu, :len(row)] += row
        else:
            Z[enu, :] += row[:maxlen]
    return Z


def generate_traces(ascad_database, new_traces_file, train_data, attack_data,
                    trace_length=700,
                    noise_level=8,  # add Gussian noise
                    clock_range=4,  # add clock jitter
                    a=5, b=3, delay_amplitude=10, shuffling_level=20):  # add random delay
    # Open the output labeled file for writing
    shutil.copy(ascad_database, new_traces_file)
    try:
        out_file = h5py.File(new_traces_file, "r+")
    except:
        print("Error: can't open HDF5 file '%s' for writing ..." %
              new_traces_file)
        sys.exit(-1)

    all_traces = np.concatenate((train_data, attack_data), axis=0)

    print("Processing traces...")
    #profiling_traces = train_data
    new_traces = addShuffling(all_traces, shuffling_level)
    new_traces = addRandomDelay(new_traces, a, b, delay_amplitude, trace_length)
    new_traces = addClockJitter(new_traces, clock_range, trace_length)
    new_traces = addGussianNoise(new_traces, noise_level)
    print("Store traces...")
    del out_file['Profiling_traces/traces']
    out_file.create_dataset('Profiling_traces/traces', data=np.array(new_traces[:50000], dtype=np.int8))
    del out_file['Attack_traces/traces']
    out_file.create_dataset('Attack_traces/traces', data=np.array(new_traces[50000:], dtype=np.int8))
    out_file.close()
    print("Done!")

if __name__ == "__main__":

    clean_trace_dir = "/home/nfs/lwu3/ASCAD/Trace/Noisy_traces/Noisy_desync_50.h5"
    noisy_traces_dir = "/home/nfs/lwu3/ASCAD/Trace/Noisy_traces/Noisy_all.h5"

    # load traces
    print('load traces...')
    (x_train_desync0, x_test_desync0) = load_ascad(clean_trace_dir)
    print(np.shape(x_train_desync0))
    print(np.shape(x_test_desync0))

    generate_traces(clean_trace_dir, noisy_traces_dir,
                    x_train_desync0, x_test_desync0)
