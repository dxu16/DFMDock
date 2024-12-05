import numpy as np

def choose_plane_basis(normal):
    # Choose an arbitrary vector that is not parallel to the normal vector
    if np.allclose(normal, [1, 0, 0]):
        arbitrary = np.array([0, 1, 0])
    else:
        arbitrary = np.array([1, 0, 0])
    
    # First basis vector: cross product of normal and arbitrary vector
    basis1 = np.cross(normal, arbitrary)
    basis1 = basis1 / np.linalg.norm(basis1)
    
    # Second basis vector: cross product of normal and first basis vector
    basis2 = np.cross(normal, basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    
    return basis1, basis2

def grid_proj_one(proj_mag, ortho_2D, min_ortho_2D, grid_size):
    grid_idx = np.floor((ortho_2D - min_ortho_2D) / grid_size).astype(int)
    grid = {}
    for idx in range(len(grid_idx)):
        idx_2D = (grid_idx[idx][0], grid_idx[idx][1])
        if idx_2D not in grid:
            grid[idx_2D] = []
        grid[idx_2D].append(proj_mag[idx])
    for key in grid:
        grid[key] = np.array(grid[key])
    return grid

def grid_proj(lig_pos, rec_pos, direction, grid_size):
    basis1, basis2 = choose_plane_basis(direction)

    lig_com = np.mean(lig_pos, axis=-2)

    lig_pos = lig_pos - lig_com
    rec_pos = rec_pos - lig_com

    lig_proj_mag = np.dot(lig_pos, direction)
    lig_proj = lig_proj_mag * direction
    lig_ortho = lig_pos - lig_proj
    lig_ortho_2D = np.concatenate(np.dot(lig_ortho, basis1), np.dot(lig_ortho, basis2), axis=-1)

    rec_proj_mag = np.dot(rec_pos, direction)
    rec_proj = rec_proj_mag * direction
    rec_ortho = rec_pos - rec_proj
    rec_ortho_2D = np.concatenate(np.dot(rec_ortho, basis1), np.dot(rec_ortho, basis2), axis=-1)

    both_ortho_2D = np.concatenate(lig_ortho_2D, rec_ortho_2D, axis=-2)
    min_ortho_2D = np.min(both_ortho_2D, axis=-2)

    # assuming no batch   
    lig_grid = grid_proj_one(lig_proj_mag, lig_ortho_2D, min_ortho_2D, grid_size)
    rec_grid = grid_proj_one(rec_proj_mag, rec_ortho_2D, min_ortho_2D, grid_size)

    return lig_grid, rec_grid

def find_max_clash(lig_pos, rec_pos, direction, grid_size):
    lig_grid, rec_grid = grid_proj(lig_pos, rec_pos, direction, grid_size)
    
    common_keys = set(lig_grid.keys()) & set(rec_grid.keys())

    clashes = []
    for key in common_keys:
        max_rec = np.amax(rec_grid[key])
        min_lig = np.amin(lig_grid[key])
        clashk = max_rec - min_lig
        if clashk > 0:
            clashes.append(clashk)
    
    return max(clashes)

def get_clash_force(lig_pos, rec_pos, grid_size, scale=1):
    lig_com = np.mean(lig_pos, axis=-2)
    rec_com = np.mean(rec_pos, axis=-2)
    if lig_com == rec_com:
        direction = np.array([1, 0, 0])
    else:
        direction = lig_com - rec_com
        direction = direction / np.linalg.norm(direction)
    lig_grid, rec_grid = grid_proj(lig_pos, rec_pos, direction, grid_size)
    
    common_keys = set(lig_grid.keys()) & set(rec_grid.keys())

    clashes = []
    for key in common_keys:
        rec_sum = np.sum(rec_grid[key])
        lig_sum = np.sum(lig_grid[key])
        clashk = rec_sum - lig_sum
        if clashk > 0:
            clashes.append(clashk)
    
    return sum(clashes) * scale * direction

def find_max_clash_cheap(lig_pos, rec_pos, direction):
    lig_com = np.mean(lig_pos, axis=-2)

    lig_pos = lig_pos - lig_com
    rec_pos = rec_pos - lig_com

    lig_proj_mag = np.dot(lig_pos, direction)
    rec_proj_mag = np.dot(rec_pos, direction)
    max_rec = np.amax(rec_proj_mag)
    min_lig = np.amin(lig_proj_mag)
    clash = max_rec - min_lig
    return clash
