def sign_flip(update, scale: float = 1.0):
    return {k: -scale * v for k, v in update.items()}
