METRIC_TO_CLASSES = {
    "av": ["NETWORK", "ADJACENT_NETWORK", "LOCAL", "PHYSICAL"],    # Attack Vector
    "ac": ["LOW", "HIGH"],                                         # Attack Complexity
    "pr": ["NONE", "LOW", "HIGH"],                                 # Privileges Required
    "ui": ["NONE", "REQUIRED"],                                    # User Interaction
    "s":  ["UNCHANGED", "CHANGED"],                                # Scope
    "c":  ["NONE", "LOW", "HIGH"],                                 # Confidentiality
    "i":  ["NONE", "LOW", "HIGH"],                                 # Integrity
    "a":  ["NONE", "LOW", "HIGH"],                                 # Availability
}

# Column positions for a CSV shaped like the sample above
METRIC_TO_COLIDX = {
    "av": 2,
    "ac": 3,
    "pr": 4,
    "ui": 5,
    "s":  6,
    "c":  7,
    "i":  8,
    "a":  9,
}


def metric_config(metric: str):
    """Return (classes, num_labels, label_position). Raises KeyError if unknown."""
    m = metric.lower()
    classes = METRIC_TO_CLASSES[m]
    label_pos = METRIC_TO_COLIDX[m]
    return classes, len(classes), label_pos