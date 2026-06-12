def model_scaling(update, scale: float = 5.0):
    return {k: scale * v for k, v in update.items()}
