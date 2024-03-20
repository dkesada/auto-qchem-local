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


def get_all_idx(atom, elements):
    """
    Return all indexes of specified atom inside elements
    """

    return [x for x in range(len(elements)) if atom in elements[x]]


def get_first_idx(atom, elements):
    """
    Return the first found element inside elements that matches atom
    """
    idx = get_all_idx(atom, elements)
    if len(idx) > 0:
        idx = idx[0]
    else:
        idx = None
    
    return idx


def get_closest_atom_to_metal(atom, elements, metal_idx, coordinates):
    """
    Return the index of the closest element that matches atom to the metal in metal_idx
    """
    atoms_idx = get_all_idx(atom, elements)
    metal_coords = coordinates[metal_idx]
    dist = [euclid_dist(metal_coords, coordinates[x]) for x in atoms_idx]

    return atoms_idx[np.argmin(dist)]


def get_dist_vector(atom_idx, coordinates):
    """
    Return the distance from the atom in the specified index to all the other elements
    """
    atom_coords = coordinates[atom_idx]

    return [euclid_dist(atom_coords, x) for x in coordinates]


def get_central_carbon(elements, coordinates, metal_idx):
    """
    Get the central carbon between a copper and two nitrogens. This will be the closest to the three of them combined.
    """

    # Get all carbon indexes
    carbon_idx = get_all_idx('C', elements)

    # Get all nitrogen indexes
    nitrogen_idx = get_all_idx('N', elements)

    # Calculate carbon distances to the three atoms
    carbon_dist = [euclid_dist(coordinates[idx], coordinates[metal_idx]) +
                   euclid_dist(coordinates[idx], coordinates[nitrogen_idx[0]]) +
                   euclid_dist(coordinates[idx], coordinates[nitrogen_idx[1]]) for idx in carbon_idx]

    # Get the closest carbon
    carbon_min_idx = np.argmin(carbon_dist)

    return carbon_idx[carbon_min_idx]

