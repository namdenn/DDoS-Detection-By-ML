import csv
import re
import statistics

# Input and output file paths
input_file = '/home/ubuntu/year3/group_project/internal_pcap1(FULL_Attacker).csv'
output_file = '/home/ubuntu/year3/group_project/extracted_flow_features.csv'

# Updated header for the output file (columns removed)
output_header = [
    'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol',
    'Duration', 'Total Packets', 'Total Length', 'Packet Length Min', 'Packet Length Max',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Flow IAT Total', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count',
    'Active Mean', 'Active Std', 'Idle Mean', 'Idle Std','Label'
]

# Regex for filtering TCP tags in the 'Info' column
tcp_tags = r'\[(TCP Retransmission|TCP Out-Of-Order|TCP Previous segment not captured)\]'
tcp_flags = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'CWR', 'ECE']

# Function to extract features from a single row
def extract_features(row):
    src_ip = row['Source']
    dst_ip = row['Destination']
    protocol = row['Protocol']
    timestamp = float(row['Time'])
    length = int(row['Length'])
    flags = {flag: int(row['Info'].count(flag)) for flag in tcp_flags}  # Count TCP flags
   
    info_example = re.sub(tcp_tags, '', row['Info']).strip() if 'Info' in row else ''
    src_port, dst_port = 'Unknown', 'Unknown'

    if ' > ' in info_example:
        parts = info_example.split(' > ')
        src_port = parts[0].strip()
        dst_port = parts[1].split()[0].strip()
    elif ' -> ' in info_example:
        parts = info_example.split(' -> ')
        src_port = parts[0].strip()
        dst_port = parts[1].split()[0].strip()
    else:
        src_port, dst_port = 0, 0

    flow_id = f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{protocol}"
    return flow_id, src_ip, src_port, dst_ip, dst_port, protocol, timestamp, length, flags

# Dictionary to store flows
flows = {}

with open(input_file, mode='r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        flow_id, src_ip, src_port, dst_ip, dst_port, protocol, timestamp, length, flags = extract_features(row)
        unique_key = (src_ip, src_port, dst_ip, dst_port, protocol)

        if unique_key not in flows:
            flows[unique_key] = {
                'flow_id': flow_id,
                'src_ip': src_ip,
                'src_port': src_port,
                'dst_ip': dst_ip,
                'dst_port': dst_port,
                'protocol': protocol,
                'start_time': timestamp,
                'end_time': timestamp,
                'total_packets': 1,
                'total_length': length,
                'min_length': length,
                'max_length': length,
                'timestamps': [timestamp],
                'flags': flags,
                'active_periods': [],
                'idle_periods': [],
                'last_timestamp': timestamp
            }
        else:
            flow = flows[unique_key]
            flow['start_time'] = min(flow['start_time'], timestamp)
            flow['end_time'] = max(flow['end_time'], timestamp)
            flow['total_packets'] += 1
            flow['total_length'] += length
            flow['min_length'] = min(flow['min_length'], length)
            flow['max_length'] = max(flow['max_length'], length)
            flow['timestamps'].append(timestamp)
            for flag, count in flags.items():
                flow['flags'][flag] += count

            # Calculate active/idle times
            active_time = timestamp - flow['last_timestamp']
            if active_time > 0:
                flow['active_periods'].append(active_time)
            else:
                flow['idle_periods'].append(-active_time)
            flow['last_timestamp'] = timestamp

# Prepare rows for writing to output
rows = []
for flow in flows.values():
    duration = flow['end_time'] - flow['start_time']
    flow_bytes_per_s = flow['total_length'] / duration if duration > 0 else 0
    flow_packets_per_s = flow['total_packets'] / duration if duration > 0 else 0

    # Calculate inter-arrival times (IATs)
    timestamps = sorted(flow['timestamps'])
    iats = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

    iat_mean = statistics.mean(iats) if iats else 0
    iat_std = statistics.stdev(iats) if len(iats) > 1 else 0
    iat_max = max(iats) if iats else 0
    iat_min = min(iats) if iats else 0
    iat_total = sum(iats)

    active_periods = flow['active_periods']
    idle_periods = flow['idle_periods']

    active_mean = statistics.mean(active_periods) if active_periods else 0
    active_std = statistics.stdev(active_periods) if len(active_periods) > 1 else 0
    idle_mean = statistics.mean(idle_periods) if idle_periods else 0
    idle_std = statistics.stdev(idle_periods) if len(idle_periods) > 1 else 0

    rows.append([
        flow['flow_id'], flow['src_ip'], flow['src_port'], flow['dst_ip'], flow['dst_port'], flow['protocol'],
        duration, flow['total_packets'], flow['total_length'], flow['min_length'], flow['max_length'],
        flow_bytes_per_s, flow_packets_per_s, iat_mean, iat_std, iat_max, iat_min, iat_total,
        flow['flags']['FIN'], flow['flags']['SYN'], flow['flags']['RST'], flow['flags']['PSH'],
        flow['flags']['ACK'], flow['flags']['URG'], flow['flags']['CWR'], flow['flags']['ECE'],
        active_mean, active_std, idle_mean, idle_std
    ])

rows.sort(key=lambda x: x[0])

with open(output_file, mode='w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(output_header)
    writer.writerows(rows)

print(f"Feature extraction completed. Results saved to '{output_file}'.")
