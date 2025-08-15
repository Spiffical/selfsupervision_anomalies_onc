# hierarchical labels structure for marine acoustic classification
# based on standardized taxonomy from the provided images

HIERARCHICAL_LABELS = {
    "Anthropophony": {
        "In-air source": {
            "Aircraft": {},
            "Snowmobile": {}
        },
        "Industrial activity": {
            "Dredging": {},
            "Mining": {},
            "Pile driving": {}
        },
        "Sonar": {
            "Fisheries sonar": {},
            "Naval sonar": {}
        },
        "Submersible": {
            "Human-occupied vehicle": {},
            "Remotely operated vehicle": {}
        },
        "Surveying": {
            "Airgun": {},
            "Explosive": {}
        },
        "Unknown anthropophony": {},
        "Vessel": {
            "Cargo ship": {},
            "Fishing": {},
            "Icebreaker": {},
            "Military ship": {},
            "Passenger ship": {},
            "Pleasure craft": {},
            "Research vessel": {},
            "Sailing": {},
            "Tanker": {},
            "Tug": {}
        }
    },
    "Biophony": {
        "Crustacean": {
            "Crab": {},
            "Lobster": {},
            "Shrimp": {
                "Snapping shrimp": {}
            }
        },
        "Fish": {
            "Vent fish": {},
            "Fish chorus": {}
        },
        "Marine mammal": {
            "Cetacean": {
                "Baleen whale": {
                    "Bowhead whale": {},
                    "Blue whale": {},
                    "Fin whale": {},
                    "Gray whale": {},
                    "Humpback whale": {},
                    "Minke whale": {},
                    "North Atlantic right whale": {},
                    "North Pacific right whale": {},
                    "Sei whale": {}
                },
                "Toothed whale": {
                    "Beaked whales": {
                        "Baird's beaked whale": {},
                        "Cuvier's beaked whale": {}
                    },
                    "Beluga": {},
                    "Dolphin": {
                        "Atlantic spotted dolphin": {},
                        "Common bottlenose dolphin": {},
                        "Common dolphin": {},
                        "Northern right whale dolphin": {},
                        "Pacific white-sided dolphin": {},
                        "Risso's dolphin": {},
                        "Striped dolphin": {}
                    },
                    "False killer whale": {},
                    "Killer whale": {
                        "Bigg's killer whale": {},
                        "Northern resident killer whale": {},
                        "Offshore killer whale": {},
                        "Southern resident killer whale": {}
                    },
                    "Narwhal": {},
                    "Porpoise": {
                        "Dall's porpoise": {},
                        "Harbour porpoise": {}
                    },
                    "Sperm whale": {}
                }
            },
            "Pinniped": {
                "Seal": {},
                "Walrus": {}
            }
        },
        "Unknown biophony": {
            "Bioacoustic communication signal": {},
            "Echolocation click": {},
            "Click train": {},
            "Drumming": {},
            "Grinding": {},
            "Snapping": {},
            "Stridulation": {},
            "Vocalization": {}
        }
    },
    "Geophony": {
        "Environmental sound": {
            "Flow noise": {},
            "Ice cracking": {},
            "Iceberg collision": {},
            "Tsunami": {}
        },
        "Geology": {
            "Bubbling": {
                "Methane seep": {}
            },
            "Earthquake": {},
            "Hydrothermal event": {
                "Chimney collapse": {},
                "Impulse": {}
            },
            "Magma": {},
            "Sedimentation": {},
            "Turbidity current": {}
        },
        "Weather": {
            "Lightning strike": {},
            "Precipitation": {
                "Hail": {},
                "Rain": {},
                "Snow": {}
            },
            "Wind": {},
            "Waves": {}
        },
        "Unknown geophony": {}
    },
    "Instrumentation": {
        "Hydrophone contact": {},
        "Malfunction": {
            "Clipping": {},
            "Data gap": {},
            "Frequency dropout": {},
            "Sensitivity change": {},
            "Time dropout": {}
        },
        "Other ONC equipment": {
            "ADCP": {},
            "Camera": {},
            "Mooring noise": {
                "Chain noise": {}
            }
        },
        "Self-noise": {
            "Acoustic self-noise": {},
            "Non-acoustic self noise": {
                "Tonal": {}
            }
        },
        "Unknown instrumentation": {}
    },
    "Other": {
        "Ambient sound": {},
        "Unknown sound of interest": {}
    }
}

def get_all_paths(hierarchy=None, current_path=None):
    """
    Get all possible label paths from the hierarchy.
    Returns a list of tuples where each tuple represents a path.
    """
    if hierarchy is None:
        hierarchy = HIERARCHICAL_LABELS
    if current_path is None:
        current_path = []
    
    paths = []
    
    for key, value in hierarchy.items():
        new_path = current_path + [key]
        paths.append(tuple(new_path))
        
        if isinstance(value, dict) and value:
            # recursively get paths from deeper levels
            deeper_paths = get_all_paths(value, new_path)
            paths.extend(deeper_paths)
    
    return paths

def get_label_display_name(path):
    """Convert a path tuple to a display-friendly string"""
    return " > ".join(path)

def get_flat_labels():
    """Get all leaf labels for backwards compatibility"""
    all_paths = get_all_paths()
    # return only the deepest level labels that don't have children
    leaf_labels = []
    
    def is_leaf(path, hierarchy=HIERARCHICAL_LABELS):
        current = hierarchy
        for part in path:
            current = current[part]
        return not isinstance(current, dict) or not current
    
    for path in all_paths:
        if is_leaf(path):
            leaf_labels.append(path[-1])  # just the final label name
    
    return leaf_labels

def path_to_string(path):
    """Convert path tuple to string representation for JSON storage"""
    return " > ".join(path)

def string_to_path(path_string):
    """Convert string representation back to path tuple"""
    return tuple(path_string.split(" > "))

# legacy flat labels mapping to hierarchical paths
LEGACY_LABEL_MAPPING = {
    "Unknown Feature": "Other > Unknown sound of interest",  # put this first so it takes priority
    "Anomaly": "Other > Unknown sound of interest",
    "Data Gap": "Instrumentation > Malfunction > Data gap", 
    "Dropout": "Instrumentation > Malfunction > Frequency dropout",
    "Engine Noise": "Anthropophony > Vessel",  # generic vessel since we don't know type
    "Rain": "Geophony > Weather > Precipitation > Rain",
    "Sensitivity": "Instrumentation > Malfunction > Sensitivity change",
    "Tonal": "Instrumentation > Self-noise > Non-acoustic self noise > Tonal",
    # add some common variations
    "Unknown Features": "Other > Unknown sound of interest",
    "Engine noise": "Anthropophony > Vessel",
    "rain": "Geophony > Weather > Precipitation > Rain",
    "tonal": "Instrumentation > Self-noise > Non-acoustic self noise > Tonal"
}

def is_legacy_format(label_data):
    """
    Check if the loaded data is in legacy flat format.
    Returns True if it looks like old format, False if hierarchical.
    """
    # sample some labels to detect format
    sample_labels = []
    for filename, labels in label_data.items():
        if isinstance(labels, list):
            sample_labels.extend(labels[:3])  # just sample a few
        else:
            sample_labels.append(labels)
        if len(sample_labels) >= 10:  # enough samples
            break
    
    hierarchical_count = 0
    legacy_count = 0
    
    for label in sample_labels:
        if isinstance(label, str):
            if " > " in label:
                hierarchical_count += 1
            elif label in LEGACY_LABEL_MAPPING:
                legacy_count += 1
    
    # if we see more legacy patterns than hierarchical, treat as legacy
    return legacy_count > hierarchical_count

def convert_legacy_to_hierarchical(legacy_labels):
    """Convert list of legacy flat labels to hierarchical format"""
    hierarchical_labels = []
    for label in legacy_labels:
        if isinstance(label, str) and label in LEGACY_LABEL_MAPPING:
            hierarchical_labels.append(LEGACY_LABEL_MAPPING[label])
        else:
            # if we can't map it, try to keep it as-is or map to unknown
            if isinstance(label, str) and " > " not in label:
                hierarchical_labels.append(f"Other > Unknown sound of interest")
            else:
                hierarchical_labels.append(str(label))
    return hierarchical_labels

def convert_hierarchical_to_legacy(hierarchical_labels):
    """Convert hierarchical labels back to legacy flat format for compatibility"""
    # create reverse mapping, prioritizing canonical forms (first occurrence)
    reverse_mapping = {}
    for legacy, hierarchical in LEGACY_LABEL_MAPPING.items():
        if hierarchical not in reverse_mapping:
            reverse_mapping[hierarchical] = legacy
    
    legacy_labels = []
    for label in hierarchical_labels:
        if isinstance(label, str):
            if label in reverse_mapping:
                legacy_labels.append(reverse_mapping[label])
            elif " > " in label:
                # extract leaf label as fallback
                leaf = label.split(" > ")[-1]
                legacy_labels.append(leaf)
            else:
                legacy_labels.append(label)
        else:
            legacy_labels.append(str(label))
    
    return legacy_labels

def is_valid_path(path, hierarchy=None):
    """Check if a given path is valid in the hierarchy"""
    if hierarchy is None:
        hierarchy = HIERARCHICAL_LABELS
    
    current = hierarchy
    for part in path:
        if part not in current:
            return False
        current = current[part]
        if not isinstance(current, dict):
            return False
    return True