import os
import csv
import librosa

# Define the root directory where the folders containing audio files are stored
root_directory = '/path/to/your/audio/files'

# Define the path for the metadata CSV file
metadata_csv_path = 'metadata.csv'

# Initialize the list to store metadata
metadata = []

# List of command folders
commands = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']

# Iterate over each command folder in the root directory
for command in commands:
    # Construct the full path to the command folder
    command_path = os.path.join(root_directory, command)
    
    # Check if it's a directory
    if os.path.isdir(command_path):
        # Iterate over each file in the command folder
        for file_name in os.listdir(command_path):
            # Check if the file is an audio file
               file_name.endswith('.wav'):
                # Get the full path of the audio file
                file_path = os.path.join(command_path, file_name)
                
                # Load the audio file to get its duration
                audio, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=audio, sr=sr)
                
                # Append the metadata for the audio file to the list
                metadata.append([file_name, duration, command])

# Write the metadata to the CSV file
with open(metadata_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(['file_name', 'duration', 'label'])
    # Write the metadata rows
    writer.writerows(metadata)

print(f"Metadata for {len(metadata)} audio files has been successfully written to {metadata_csv_path}.")
