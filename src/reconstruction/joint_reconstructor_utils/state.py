"""Release temporary operators and select state for the next resolution."""


def remove_temporary_operators(data):
    """Remove operator references in place and return the same dictionary.
    """
    if data is None:
        return None
    for key in ("MotionOperator", "E", "J"):
        data.pop(key, None)
    return data


def extract_image_and_motion_for_next_level(data):
    """Return the grid dimensions and estimates used to initialize the next level.
    """
    if data is None:
        return None
    return {
        "Nx": data["Nx"],
        "Ny": data["Ny"],
        "Nz": data.get("Nz", 1),
        "ReconstructedImage": data["ReconstructedImage"],
        "MotionModel": data["MotionModel"],
    }
