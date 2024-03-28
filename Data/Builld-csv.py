import os
import csv

# Data parameters
data_dir = "path/to/your/data/directory"  # Replace with the actual path
commands = ["power on focus", "power off focus", "light on", "light off", "one", "two", "three", "four"]

# CSV file creation
with open("audio_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "command"])  # Header row

    for command_index, command in enumerate(commands):
        command_path = os.path.join(data_dir, command)
        for filename in os.listdir(command_path):
            audio_path = os.path.join(command_path, filename)
            writer.writerow([filename, command])  # Write filename and command to CSV
