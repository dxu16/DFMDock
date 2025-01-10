import torch

def choose_plane_basis(normal):
    # Choose an arbitrary vector that is not parallel to the normal vector
    if torch.allclose(normal, torch.tensor([1, 0, 0], dtype=torch.float32).to(normal.device)):
        arbitrary = torch.tensor([0, 1, 0], dtype=torch.float32).to(normal.device)
    else:
        arbitrary = torch.tensor([1, 0, 0], dtype=torch.float32).to(normal.device)
    
    # First basis vector: cross product of normal and arbitrary vector
    basis1 = torch.cross(normal, arbitrary)
    basis1 = basis1 / torch.norm(basis1)
    
    # Second basis vector: cross product of normal and first basis vector
    basis2 = torch.cross(normal, basis1)
    basis2 = basis2 / torch.norm(basis2)
    
    return basis1, basis2

def grid_proj_one(proj_mag, ortho_2D, min_ortho_2D, grid_size):
    grid_idx = torch.floor((ortho_2D - min_ortho_2D) / grid_size).int()
    grid = {}
    for idx in range(len(grid_idx)):
        idx_2D = (grid_idx[idx][0].item(), grid_idx[idx][1].item())
        if idx_2D not in grid:
            grid[idx_2D] = []
        grid[idx_2D].append(proj_mag[idx])
    for key in grid:
        grid[key] = torch.tensor(grid[key])
    return grid

def grid_proj(lig_pos, rec_pos, direction, grid_size):
    basis1, basis2 = choose_plane_basis(direction)

    lig_pos_ca = lig_pos[..., 1, :]
    rec_pos_ca = rec_pos[..., 1, :]

    lig_com = torch.mean(lig_pos_ca, dim=-2)

    lig_pos_ca = lig_pos_ca - lig_com
    rec_pos_ca = rec_pos_ca - lig_com

    lig_proj_mag = torch.tensordot(lig_pos_ca, direction, dims=1)
    lig_proj = lig_proj_mag.unsqueeze(-1) * direction.unsqueeze(-2)
    lig_ortho = lig_pos_ca - lig_proj
    lig_ortho_2D = torch.cat((torch.tensordot(lig_ortho, basis1, dims=1).unsqueeze(-1), 
                              torch.tensordot(lig_ortho, basis2, dims=1).unsqueeze(-1)), axis=-1)

    rec_proj_mag = torch.tensordot(rec_pos_ca, direction, dims=1)
    rec_proj = rec_proj_mag.unsqueeze(-1) * direction.unsqueeze(-2)
    rec_ortho = rec_pos_ca - rec_proj
    rec_ortho_2D = torch.cat((torch.tensordot(rec_ortho, basis1, dims=1).unsqueeze(-1), 
                              torch.tensordot(rec_ortho, basis2, dims=1).unsqueeze(-1)), axis=-1)

    both_ortho_2D = torch.cat((lig_ortho_2D, rec_ortho_2D), axis=-2)
    min_ortho_2D = torch.amin(both_ortho_2D, axis=-2)

    # assuming no batch   
    lig_grid = grid_proj_one(lig_proj_mag, lig_ortho_2D, min_ortho_2D, grid_size)
    rec_grid = grid_proj_one(rec_proj_mag, rec_ortho_2D, min_ortho_2D, grid_size)

    return lig_grid, rec_grid

def find_max_clash(lig_pos, rec_pos, direction, grid_size):
    lig_grid, rec_grid = grid_proj(lig_pos, rec_pos, direction, grid_size)
    
    common_keys = set(lig_grid.keys()) & set(rec_grid.keys())

    clashes = []
    for key in common_keys:
        max_rec = torch.amax(rec_grid[key])
        min_lig = torch.amin(lig_grid[key])
        clashk = max_rec - min_lig
        if clashk > 0:
            clashes.append(clashk)
    
    return max(clashes)

def get_clash_force(lig_pos, rec_pos, grid_size, scale=1):
    lig_com = torch.mean(lig_pos, axis=-2)
    rec_com = torch.mean(rec_pos, axis=-2)
    if lig_com == rec_com:
        direction = torch.array([1, 0, 0]).to(lig_pos.device)
    else:
        direction = lig_com - rec_com
        direction = direction / torch.linalg.norm(direction)
    lig_grid, rec_grid = grid_proj(lig_pos, rec_pos, direction, grid_size)
    
    common_keys = set(lig_grid.keys()) & set(rec_grid.keys())

    clashes = []
    for key in common_keys:
        rec_sum = torch.sum(rec_grid[key]).item()
        lig_sum = torch.sum(lig_grid[key]).item()
        clashk = rec_sum - lig_sum
        if clashk > 0:
            clashes.append(clashk)
    
    return sum(clashes) * scale * direction

def find_max_clash_cheap(lig_pos, rec_pos, direction):
    lig_com = torch.mean(lig_pos, axis=-2)

    lig_pos = lig_pos - lig_com
    rec_pos = rec_pos - lig_com

    lig_proj_mag = torch.dot(lig_pos, direction)
    rec_proj_mag = torch.dot(rec_pos, direction)
    max_rec = torch.amax(rec_proj_mag)
    min_lig = torch.amin(lig_proj_mag)
    clash = max_rec - min_lig
    return clash
