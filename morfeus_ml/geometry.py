import numpy as np
# from scipy.spatial.distance import cdist


def euclid_dist(p1, p2):
    """ 
    Euclidean distance between two points in the x y z dimension 
    :param numpy.ndarray p1: first point coordinates [x, y, z, ...]
    :param numpy.ndarray p2: second point coordinates [x, y, z, ...]
    :return: the Euclidean distance between both points
    """
    # cdist(np.array(p1).reshape(-1, 3), np.array(p2).reshape(-1, 3))
    return np.sqrt(sum(np.square(p1 - p2)))


def get_first_idx(atom, elements):
    """
    Return the first found element inside elements that matches atom
    """
    idx = [x for x in range(len(elements)) if atom in elements[x]]
    if len(idx) > 0:
        idx = idx[0]
    else:
        idx = None
    
    return idx


def get_closest_atom_to_metal(atom, elements, metal_idx, coordinates):
    """
    Return the index of the closest element that matches atom to the metal in metal_idx
    """
    atoms_idx = [x for x in range(len(elements)) if atom in elements[x]]
    metal_coords = coordinates[metal_idx]
    dist = [euclid_dist(metal_coords, coordinates[x]) for x in atoms_idx]

    return atoms_idx[np.argmin(dist)]
