import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import gzip
import pickle
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation 
from pinder.core import PinderSystem, get_index, get_supplementary_data, get_metadata
from pinder.data.plot.performance import get_subsampled_train
from utils import residue_constants
from biotite.structure import get_residues, get_residue_starts
from pinder.core.structure.atoms import resn2seq

logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------
# Helper functions

def prefilter(train_index, index_meta, entity_meta, chain_meta):
    # Remove entries with more than 1 consecutive 'X' in the sequence
    entity_meta['part_id'] = entity_meta['entry_id'].astype(str) + '_' + entity_meta['chain'].astype(str)
    rec_id = train_index['holo_R_pdb'].apply(lambda x: x.split('.pdb')[0])
    rec_id = rec_id.apply(lambda x: x.split('_')[0] + '_' + x.split('_')[2])
    rec_order_df = pd.DataFrame({'part_id': rec_id.tolist(), 'position': range(len(rec_id))})
    ordered_rec_data = pd.merge(rec_order_df, entity_meta, on='part_id').sort_values('position').drop('position', axis=1)
    lig_id = train_index['holo_L_pdb'].apply(lambda x: x.split('.pdb')[0])
    lig_id = lig_id.apply(lambda x: x.split('_')[0] + '_' + x.split('_')[2])
    lig_order_df = pd.DataFrame({'part_id': lig_id.tolist(), 'position': range(len(lig_id))})
    ordered_lig_data = pd.merge(lig_order_df, entity_meta, on='part_id').sort_values('position').drop('position', axis=1)
    rec_seq = ordered_rec_data.sequence
    lig_seq = ordered_lig_data.sequence
    rec_has_XX = rec_seq.apply(lambda x: 'XX' in x)
    lig_has_XX = lig_seq.apply(lambda x: 'XX' in x)
    entity_meta.drop('part_id', axis=1, inplace=True)

    # Remove entries with mismatched sequence length and auth resi number
    chain_meta_train = pd.merge(train_index, chain_meta, how='left', on='id')
    rec_seq_len = ordered_rec_data.sequence.apply(lambda x: len(x)).to_numpy()
    lig_seq_len = ordered_lig_data.sequence.apply(lambda x: len(x)).to_numpy()
    rec_resi_auth_len = chain_meta_train['resi_auth_R'].apply(lambda x: len(x.split(','))).to_numpy()
    lig_resi_auth_len = chain_meta_train['resi_auth_L'].apply(lambda x: len(x.split(','))).to_numpy()
    rec_len_mismatched = rec_seq_len != rec_resi_auth_len
    lig_len_mismatched = lig_seq_len != lig_resi_auth_len

    # Remove entries with buried_sasa less than 400
    train_with_meta = pd.merge(train_index,index_meta[["id",'buried_sasa']], how="left", on="id")
    low_buried_sasa = train_with_meta['buried_sasa'] < 400

    train_index = train_index[~rec_len_mismatched & ~lig_len_mismatched & ~rec_has_XX & ~lig_has_XX & ~low_buried_sasa]
    train_index = train_index.reset_index(drop=True)

    return train_index

def get_seq_from_atom_array(atom_array):
    pdb_res_num, pdb_res_name = get_residues(atom_array)
    part_seq = resn2seq(pdb_res_name)
    return part_seq, pdb_res_num

def get_pos(atom_array):
    resi_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
    n_res = len(resi_starts) - 1
    bb_coords = np.zeros((n_res, 3, 3)) 
    bb_mask = np.zeros((n_res, 3), dtype=bool)

    for idx in range(n_res):
        res_atoms = atom_array[resi_starts[idx]:resi_starts[idx + 1]]
        n_atom = res_atoms[res_atoms.atom_name == "N"]
        ca_atom = res_atoms[res_atoms.atom_name == "CA"]
        c_atom = res_atoms[res_atoms.atom_name == "C"]
        if len(n_atom) != 0:
            bb_coords[idx, 0, :] = n_atom.coord[0]
            bb_mask[idx, 0] = True
        if len(ca_atom) != 0:
            bb_coords[idx, 1, :] = ca_atom.coord[0]
            bb_mask[idx, 1] = True
        if len(c_atom) != 0:
            bb_coords[idx, 2, :] = c_atom.coord[0]
            bb_mask[idx, 2] = True
    
    return bb_coords, bb_mask

def load_dict_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        dict_data = pickle.load(f)
    return dict_data

def random_rotation(rec_pos, lig_pos):
    rot = torch.from_numpy(Rotation.random().as_matrix()).float()
    pos = torch.cat([rec_pos, lig_pos], dim=0)
    cen = pos.mean(dim=(0, 1))
    pos = (pos - cen) @ rot.T
    rec_pos_out = pos[:rec_pos.size(0)]
    lig_pos_out = pos[rec_pos.size(0):]
    return rec_pos_out, lig_pos_out

#----------------------------------------------------------------------------
# Dataset class

class PinderDataset(Dataset):
    def __init__(
        self, 
        test_split: str = 'pinder_s',
        mode: str = 'train',
        use_esm: bool = False,
    ):
        self.mode = mode
        self.use_esm = use_esm

        # Load the dictionary data
        chain_meta = get_supplementary_data("chain_metadata")
        full_index = get_index()
        if self.mode == 'train':
            train_index = full_index.query("split == 'train'").copy().reset_index(drop=True)
            train_index = prefilter(train_index,
                                    get_metadata(),
                                    get_supplementary_data("entity_metadata"),
                                    chain_meta)
            self.data_index = get_subsampled_train(train_index)
        elif self.mode == 'val':
            self.data_index = full_index.query("split == 'val'").copy().reset_index(drop=True)
        elif self.mode == 'test':
            self.data_index = full_index.query(f'{test_split} == True').copy().reset_index(drop=True)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
        self.data_index = self.data_index.merge(chain_meta[['id','resi_auth_L','resi_auth_R', 'resi_pdb_L', 'resi_pdb_R']], on='id', how='left')

        if self.use_esm:
            self.esm_cache = Path('/scratch4/jgray21/dxu39/data/pinder/2024-02/esm_c_cache')

    def __getitem__(self, idx: int):
        index_entry = self.data_index.iloc[idx]
        struct_id = index_entry['id']
        try:
            ps = PinderSystem(struct_id)

            pos_dict = {}
            pos_mask_dict = {}
            seq_dict = {}
            esm_embedding_dict = {}
            for part in ['rec', 'lig']:
                abbr = 'R' if part == 'rec' else 'L'
                part_id = index_entry[f'holo_{abbr}_pdb'].split('.pdb')[0]
                part_resi_auth = index_entry[f"resi_auth_{abbr}"]
                part_resi_auth_split = part_resi_auth.strip(',').split(',')
                resolved_index = [idx for idx, resi in enumerate(part_resi_auth_split) if resi != '']
                part_seq, part_struct_res_num = get_seq_from_atom_array(getattr(ps, f'native_{abbr}').atom_array)
                assert len(resolved_index) == len(part_struct_res_num) == len(part_seq), "Mismatched lengths between resi auth and structure"
                part_pos, part_mask = get_pos(getattr(ps, f'native_{abbr}').atom_array)
                assert len(part_seq) == part_pos.shape[0], "Mismatched pos and seq length"
                part_has_ca = np.where(part_mask[:, 1] == True)[0]
                part_pos = part_pos[part_has_ca, :, :]
                part_mask = part_mask[part_has_ca, :]
                part_seq = ''.join(part_seq[i] for i in part_has_ca)
                if self.use_esm:
                    part_esm_embedding = np.load(self.esm_cache / f"{part_id}.npy")
                    part_esm_embedding = part_esm_embedding[resolved_index, :]
                    part_esm_embedding = part_esm_embedding[part_has_ca, :]
                    assert part_esm_embedding.shape[0] == len(part_seq), "Mismatched esm embedding and seq length"
                    esm_embedding_dict[part] = part_esm_embedding
                pos_dict[part] = part_pos
                pos_mask_dict[part] = part_mask
                seq_dict[part] = part_seq
            
            _id = struct_id
            rec_seq = seq_dict['rec']
            lig_seq = seq_dict['lig']
            rec_pos = torch.from_numpy(pos_dict['rec']).float()
            lig_pos = torch.from_numpy(pos_dict['lig']).float()
            rec_pos_mask = torch.from_numpy(pos_mask_dict['rec'])
            lig_pos_mask = torch.from_numpy(pos_mask_dict['lig'])

            # One-Hot embeddings
            rec_x = torch.from_numpy(residue_constants.sequence_to_onehot(
                sequence=rec_seq,
                mapping=residue_constants.restype_order_with_x,
                map_unknown_to_x=True,
            )).float()

            lig_x = torch.from_numpy(residue_constants.sequence_to_onehot(
                sequence=lig_seq,
                mapping=residue_constants.restype_order_with_x,
                map_unknown_to_x=True,
            )).float()

            # ESM embeddings
            if self.use_esm:
                rec_esm = torch.from_numpy(esm_embedding_dict['rec']).float()
                lig_esm = torch.from_numpy(esm_embedding_dict['lig']).float()

                rec_x = torch.cat([rec_esm, rec_x], dim=-1)
                lig_x = torch.cat([lig_esm, lig_x], dim=-1)

            if self.mode == 'train':
                # shuffle the order of rec and lig
                vars_list = [(rec_x, rec_pos), (lig_x, lig_pos)]
                random.shuffle(vars_list)
                rec_x, rec_pos = vars_list[0]
                lig_x, lig_pos = vars_list[1]

            # random rotation augmentation
            rec_pos, lig_pos = random_rotation(rec_pos, lig_pos)

            # is homomer
            is_homomer = rec_seq == lig_seq

            # Output
            output = {
                'id': _id,
                'rec_seq': rec_seq,
                'lig_seq': lig_seq,
                'rec_x': rec_x,
                'lig_x': lig_x,
                'rec_pos': rec_pos,
                'lig_pos': lig_pos,
                'is_homomer': is_homomer,
                'rec_pos_mask': rec_pos_mask,
                'lig_pos_mask': lig_pos_mask,
            }
            
            return {key: value for key, value in output.items()}
        
        except Exception as e:
            if self.mode == 'train' or self.mode == 'val':
                new_idx = torch.randint(0, len(self.data_index), (1,)).item()
                logger.warning(f"Error loading {struct_id}: {e}, trying to replace with {self.data_index.iloc[new_idx]['id']}")
                return self[new_idx]
            else:
                logger.error(f"Error loading {struct_id}: {e}")
                raise e

    def __len__(self):
        return len(self.data_index)


#----------------------------------------------------------------------------
# DataModule class

class PinderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        use_esm: bool = True,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.use_esm = use_esm
        self.num_workers = kwargs['num_workers']
        self.pin_memory = kwargs['pin_memory']

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
    
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.data_train = PinderDataset(
            use_esm=self.use_esm,
            mode='train',
        )
        self.data_val = PinderDataset(
            use_esm=self.use_esm,
            mode='val',
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

#----------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    dataset = PinderDataset(
        test_split='pinder_s',
        mode='train',
        use_esm=True,
    )
    print(dataset[0])
