def label_flip_targets(y, num_classes: int = 10):
    return (num_classes - 1) - y
