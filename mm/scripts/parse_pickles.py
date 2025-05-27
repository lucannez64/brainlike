"""
Script to parse neuronal pickle simulation files into CSV for Rust ingestion.
Each pickle contains 128 simulations of shape (n_timesteps, n_features).
Outputs one CSV per pickle in the same folder with `.csv` extension.
"""
import os
import pickle
import numpy as np
import csv
import sys

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'neuronaldata'))
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(OUT_DIR, exist_ok=True)

def dict2bin(spike_map, num_segments, sim_duration_ms):
    buf = np.zeros((num_segments, sim_duration_ms), dtype=float)
    for seg, times in spike_map.items():
        idx = int(seg)
        for t in times:
            it = int(t)
            if 0 <= it < sim_duration_ms:
                buf[idx, it] = 1.0
    return buf

def parse_and_stream(sim_file, out_csv):
    # Load pickle dict
    exp = pickle.load(open(sim_file, 'rb'), encoding='latin1')
    params = exp['Params']
    results = exp['Results']['listOfSingleSimulationDicts']
    num_segments = len(params['allSegmentsType'])
    sim_duration_ms = int(params['totalSimDurationInSec'] * 1000)
    # open CSV writer
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # for each simulation and time step, write one row
        for sim_id, sim_dict in enumerate(results):
            ex_map = sim_dict['exInputSpikeTimes']
            inh_map = sim_dict['inhInputSpikeTimes']
            soma = sim_dict['somaVoltageLowRes']
            spikes = (np.array(sim_dict['outputSpikeTimes'], dtype=float) - 0.5).astype(int)
            # pre-bin spikes into set for quick lookup
            spike_set = set(int(t) for t in spikes if 0 <= t < sim_duration_ms)
            for t in range(sim_duration_ms):
                # features: excitatory followed by inhibitory
                row_feats = []
                for seg in range(num_segments):
                    row_feats.append(1.0 if t in ex_map.get(seg, []) else 0.0)
                for seg in range(num_segments):
                    row_feats.append(1.0 if t in inh_map.get(seg, []) else 0.0)
                y_reg = float(soma[t])
                y_clf = 1.0 if t in spike_set else 0.0
                writer.writerow([sim_id, t] + row_feats + [y_reg, y_clf])
    print(f"Written {out_csv}")
    
# entrypoint uses parse_and_stream instead of building full arrays
if __name__ == '__main__':
    # Determine .p files to parse
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.p')]
    for sim_file in files:
        out_csv = os.path.join(OUT_DIR, os.path.basename(sim_file).replace('.p', '.csv'))
        parse_and_stream(sim_file, out_csv)
