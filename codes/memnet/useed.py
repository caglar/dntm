from os import

def update_seed(filep, seed_val=None):
    if not seed_val:
        raise ValueError("seed should not be empty!")

    fh = open(filep, "wb")

