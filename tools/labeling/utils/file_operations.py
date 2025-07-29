import json
import os
from filelock import FileLock
import tempfile

# Create a FileLock instance (place this at module level)
lock_file = os.path.join(tempfile.gettempdir(), 'labels_lock.lock')
file_lock = FileLock(lock_file)

def load_labels(filename):
    """Load labels from JSON file with support for both old and new formats"""
    if not os.path.exists(filename):
        return {}
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert old format to new format for consistency
    converted_data = {}
    for file_key, labels in data.items():
        if isinstance(labels, list):
            converted_labels = []
            for label in labels:
                if isinstance(label, str):
                    # check if it's already hierarchical format
                    if " > " in label:
                        converted_labels.append(label)
                    else:
                        # old format - keep as single item for backward compatibility
                        converted_labels.append(label)
                else:
                    # handle other formats if needed
                    converted_labels.append(str(label))
            converted_data[file_key] = converted_labels
        else:
            # handle non-list formats
            converted_data[file_key] = [str(labels)]
    
    return converted_data


def convert_old_labels_to_hierarchical(old_labels):
    """
    Convert old flat labels to hierarchical format where possible.
    This function attempts to map old labels to the new hierarchy.
    """
    from hierarchical_labels import HIERARCHICAL_LABELS, get_all_paths, path_to_string
    
    # get all possible paths and create a mapping from leaf labels
    all_paths = get_all_paths()
    leaf_to_path_map = {}
    
    for path in all_paths:
        leaf_label = path[-1].lower()
        # only map if it's actually a leaf (no children)
        current = HIERARCHICAL_LABELS
        for part in path:
            current = current[part]
        if not isinstance(current, dict) or not current:
            leaf_to_path_map[leaf_label] = path
    
    converted_labels = []
    for label in old_labels:
        label_lower = label.lower()
        
        # direct mapping from old labels to new hierarchy
        if label_lower in leaf_to_path_map:
            converted_labels.append(path_to_string(leaf_to_path_map[label_lower]))
        elif label_lower == "rain":
            converted_labels.append("Geophony > Weather > Precipitation > Rain")
        elif label_lower == "engine noise":
            converted_labels.append("Anthropophony > Vessel")  # generic vessel since we don't know the type
        elif label_lower == "biological sounds":
            converted_labels.append("Biophony")  # generic biophony
        elif label_lower == "ambient noise":
            converted_labels.append("Other > Ambient sound")
        elif label_lower == "unknown features":
            converted_labels.append("Other > Unknown sound of interest")
        else:
            # if we can't map it, keep as is for now
            converted_labels.append(label)
    
    return converted_labels


def get_backward_compatible_labels(hierarchical_labels):
    """
    Convert hierarchical labels back to simple labels for backward compatibility
    with old dataset processing code.
    """
    flat_labels = []
    for label in hierarchical_labels:
        if " > " in label:
            # extract the leaf label from hierarchical path
            leaf_label = label.split(" > ")[-1]
            flat_labels.append(leaf_label)
        else:
            # already flat
            flat_labels.append(label)
    
    return flat_labels


def save_labels(output_file, label_data, remove=False):
    with file_lock:
        # First read existing data
        current_data = {}
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                current_data = json.load(f)
        
        # Process the label data - support both old and new formats
        for filename, labels in label_data.items():
            if remove:
                # Remove the specified labels
                if filename in current_data:
                    labels_to_remove = [labels] if not isinstance(labels, list) else labels
                    
                    # Handle both old format (strings) and new format (list of paths)
                    if isinstance(current_data[filename], list):
                        # Check if it's old format (list of strings) or new format (list of paths)
                        if current_data[filename] and isinstance(current_data[filename][0], str) and " > " not in current_data[filename][0]:
                            # Old format - simple string matching
                            current_data[filename] = [
                                label for label in current_data[filename] 
                                if str(label) not in map(str, labels_to_remove)
                            ]
                        else:
                            # New hierarchical format - path matching
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