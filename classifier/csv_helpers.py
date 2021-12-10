import csv

NUM_POINTS = 33

def write_headers():
    headers = ["class"]
    for i in range(NUM_POINTS):
        headers.extend([f"x{i+1}", f"y{i+1}", f"z{i+1}", f"v{i+1}"])
    with open('train_data.csv', 'w+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(headers)

def write_vector(label, feature_vector):
    row = [label]
    row.extend(feature_vector)
    with open('train_data.csv', 'a+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(row)