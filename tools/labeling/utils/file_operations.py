import json
import os
from filelock import FileLock
import tempfile

# Create a FileLock instance (place this at module level)
lock_file = os.path.join(tempfile.gettempdir(), 'labels_lock.lock')
file_lock = FileLock(lock_file)

def load_labels(output_file):
    with file_lock:
        labeled_files = {}
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                labeled_files = json.load(f)
        return labeled_files

def save_labels(output_file, label_data, remove=False):
    with file_lock:
        # First read existing data
        current_data = {}
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                current_data = json.load(f)
        
        # Merge new labels with existing data
        for filename, labels in label_data.items():
            if remove:
                # Remove the specified labels
                if filename in current_data:
                    labels_to_remove = [labels] if not isinstance(labels, list) else labels
                    current_data[filename] = [
                        label for label in current_data[filename] 
                        if str(label) not in map(str, labels_to_remove)
                    ]
                    # Remove the file entry if no labels remain
                    if not current_data[filename]:
                        del current_data[filename]
            else:
                # Add new labels
                if filename in current_data:
                    existing_labels = set(map(str, current_data[filename]))
                    new_labels = [labels] if not isinstance(labels, list) else labels
                    
                    for label in new_labels:
                        if str(label) not in existing_labels:
                            current_data[filename].append(label)
                else:
                    current_data[filename] = [labels] if not isinstance(labels, list) else labels
        
        # Write back the merged data
        with open(output_file, 'w') as f:
            json.dump(current_data, f, indent=4, sort_keys=True)