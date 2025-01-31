import copy
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.utils import data
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig
from datasets.docking_dataset import DockingDataset
from models.egnn_net_flow import EGNN_Net
from utils.geometry import axis_angle_to_matrix
from utils.crop import get_crop_idxs, get_crop, get_position_matrix
from utils.loss import distogram_loss

#----------------------------------------------------------------------------
# Main wrapper for training the model

class FlowMatchingDock(pl.LightningModule):
    def __init__(
        self,
        model,
        experiment,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = experiment.lr
        self.weight_decay = experiment.weight_decay

        # crop size
        self.crop_size = experiment.crop_size

        # confidence model
        self.use_confidence_loss = experiment.use_confidence_loss

        # dist model
        self.use_dist_loss = experiment.use_dist_loss

        # interface residue model
        self.use_interface_loss = experiment.use_interface_loss

        # energy
        self.grad_energy = experiment.grad_energy
        self.separate_energy_loss = experiment.separate_energy_loss
        self.use_contrastive_loss = experiment.use_contrastive_loss
        
        # translation
        self.perturb_tr = experiment.perturb_tr
        self.separate_tr_loss = experiment.separate_tr_loss

        # rotation
        self.perturb_rot = experiment.perturb_rot
        self.separate_rot_loss = experiment.separate_rot_loss

        self.use_v_loss = experiment.use_v_loss

        # # diffuser
        # if self.perturb_tr:
        #     self.r3_diffuser = R3Diffuser(diffuser.r3)
        # if self.perturb_rot:
        #     self.so3_diffuser = SO3Diffuser(diffuser.so3)

        self.tr_sigma_max = experiment.tr_sigma_max
        self.tr_sigma_min = experiment.tr_sigma_min
        self.rot_sigma_max = experiment.rot_sigma_max
        self.rot_sigma_min = experiment.rot_sigma_min

        self.restricted_perturb_tr = experiment.restricted_perturb_tr
        self.com_vec_rot_sigma = experiment.com_vec_rot_sigma

        self.scale_pred_by_t = experiment.scale_pred_by_t
        self.pred_distance = experiment.pred_distance
        self.scale_f_norm = experiment.scale_f_norm
        self.scale_tr_loss_by_sigma_max = experiment.scale_tr_loss_by_sigma_max
        self.scale_tr_loss_by_sigma_t = experiment.scale_tr_loss_by_sigma_t
        self.use_gt_v_for_e = experiment.use_gt_v_for_e

        # net
        self.net = EGNN_Net(model)
    
    def forward(self, batch):
        # move lig center to origin
        self.move_to_lig_center(batch)

        # predict
        outputs = self.net(batch, predict=True)

        return outputs

    def loss_fn(self, batch, eps=1e-5):
        with torch.no_grad():
            # uniformly sample a timestep
            t = torch.rand(1, device=self.device) * (1. - eps) + eps
            batch["t"] = t

            # sample perturbation for translation and rotation
            if self.perturb_rot:
                # rot_score_scale = self.so3_diffuser.score_scaling(t.item())
                # rot_update, rot_score_gt = self.so3_diffuser.forward_marginal(t.item())
                axis = np.random.randn(1,3)
                axis = axis / np.linalg.norm(axis)
                angle_1 = np.random.randn(1) * (self.rot_sigma_max - self.rot_sigma_min)
                angle = angle_1 * t.item()
                rot_1 = axis * angle_1
                rot_update = axis * angle
                rot_1 = torch.from_numpy(rot_1).float().to(self.device)
                rot_update = torch.from_numpy(rot_update).float().to(self.device)
                # rot_score_gt = torch.from_numpy(rot_score_gt).float().to(self.device)
            else:
                rot_update = np.zeros(3)
                rot_update = torch.from_numpy(rot_update).float().to(self.device)

            if self.perturb_tr:
                if self.restricted_perturb_tr:
                    com_lig = batch["lig_pos"].mean(dim=-2)
                    com_rec = batch["rec_pos"].mean(dim=-2)
                    com_vec = com_lig - com_rec
                    vec_rot_ang = np.random.randn(1) * self.com_vec_rot_sigma
                    vec_rot_axis = np.random.randn(1,3)
                    vec_rot_axis = vec_rot_axis / np.linalg.norm(vec_rot_axis)
                    vec_rot = vec_rot_axis * vec_rot_ang
                    vec_rot = torch.from_numpy(vec_rot).float().to(self.device)
                    tr_dir_vec = com_vec @ axis_angle_to_matrix(vec_rot).T
                    tr_mag = abs(np.random.randn() * (self.tr_sigma_max - self.tr_sigma_min))
                    tr_1 = tr_dir_vec * tr_mag
                    tr_update = tr_1 * t
                else:
                    # tr_score_scale = self.r3_diffuser.score_scaling(t.item())
                    # tr_update, tr_score_gt = self.r3_diffuser.forward_marginal(t.item())
                    tr_1 = np.random.randn(1,3) * (self.tr_sigma_max - self.tr_sigma_min)
                    tr_update = tr_1 * t.item()
                    tr_1 = torch.from_numpy(tr_1).float().to(self.device)
                    tr_update = torch.from_numpy(tr_update).float().to(self.device)
                    # tr_score_gt = torch.from_numpy(tr_score_gt).float().to(self.device)
            else:
                tr_1 = np.zeros(3)
                tr_update = np.zeros(3)
                tr_1 = torch.from_numpy(tr_1).float().to(self.device)
                tr_update = torch.from_numpy(tr_update).float().to(self.device)
            
            if self.pred_distance:
                gt_rot = -rot_update
                gt_tr = -tr_update
            else:
                gt_rot = -rot_1
                gt_tr = -tr_1

            # save gt state
            batch_gt = copy.deepcopy(batch)

            # get crop_idxs
            crop_idxs = get_crop_idxs(batch_gt, crop_size=self.crop_size)
            
            # pre crop
            batch = get_crop(batch, crop_idxs)
            batch_gt = get_crop(batch_gt, crop_idxs)

            # noised pose          
            batch["lig_pos"] = self.modify_coords(batch["lig_pos"], rot_update, tr_update)

            # get LRMSD
            l_rmsd = get_rmsd(batch["lig_pos"][..., 1, :], batch_gt["lig_pos"][..., 1, :])

            # get vt_gt
            lig_pos_pert = batch["lig_pos"] - batch["rec_pos"].mean(dim=(0, 1))
            lig_pos_gt = batch_gt["lig_pos"] - batch_gt["rec_pos"].mean(dim=(0, 1))
            vt_gt = lig_pos_gt[:, 1, :] - lig_pos_pert[:, 1, :]

            # move lig center to origin
            self.move_to_lig_center(batch)
            self.move_to_lig_center(batch_gt)

            # post crop
            #batch = get_crop(batch, crop_idxs)
            #batch_gt = get_crop(batch_gt, crop_idxs)
        
        # predict score based on the current state
        if self.grad_energy:
            outputs = self.net(batch)

            # grab some outputs
            tr_pred = outputs["tr_pred"]
            rot_pred = outputs["rot_pred"]
            v = outputs["v"]
            dedx = outputs["dedx"]
            energy_noised = outputs["energy"]

            # energy conservation loss
            if self.separate_energy_loss:
                if not self.use_gt_v_for_e:
                    v_norm = torch.norm(v, dim=-1, keepdim=True)
                    v_axis = v / (v_norm + 1e-6)
                    if self.scale_pred_by_t or self.pred_distance:
                        f_norm = v_norm
                    else:
                        f_norm = v_norm * t
                else:
                    v_norm = torch.norm(vt_gt, dim=-1, keepdim=True)
                    v_axis = vt_gt / (v_norm + 1e-6)
                    f_norm = v_norm
                if self.scale_f_norm == "DSigmoid":
                    f_norm = torch.sigmoid(f_norm) * (1 - torch.sigmoid(f_norm))
                elif self.scale_f_norm == "ISigmoid_scale":
                    f_norm = f_norm * torch.sigmoid(-f_norm) * 2
                elif self.scale_f_norm == "tanh":
                    f_norm = F.tanh(f_norm / 2)
                if self.scale_f_norm == "div_sigma_max":
                    f_norm = f_norm / self.tr_sigma_max
                if self.scale_f_norm == "div_sigma_t":
                    f_norm = f_norm / (t * self.tr_sigma_max + (1 - t) * self.tr_sigma_min)
                elif self.scale_f_norm == "none":
                    pass
                else:
                    raise ValueError(f"Unknown scale_f_norm: {self.scale_f_norm}")

                f_axis = v_axis

                dedx_norm = torch.norm(dedx, dim=-1, keepdim=True)
                dedx_axis = dedx / (dedx_norm + 1e-6)

                ec_axis_loss = torch.mean((f_axis - dedx_axis)**2)
                ec_angle_loss = torch.mean((f_norm - dedx_norm)**2)
                ec_loss = 0.5 * (ec_axis_loss + ec_angle_loss)
                
            else:
                if self.scale_pred_by_t or self.pred_distance:
                    e_grad = v
                else:
                    e_grad = v * t
                ec_loss = torch.mean((dedx - e_grad)**2)
        else:
            outputs = self.net(batch, predict=True)

            # grab some outputs
            tr_pred = outputs["tr_pred"]
            rot_pred = outputs["rot_pred"]
            energy_noised = outputs["energy"]
            
            # energy conservation loss
            ec_loss = torch.tensor(0.0, device=self.device)

        # mse_loss_fn = nn.MSELoss()
        # translation loss
        if self.perturb_tr:
            if self.scale_tr_loss_by_sigma_max:
                tr_loss_scaling_factor = self.tr_sigma_max ** 2
            elif self.scale_tr_loss_by_sigma_t:
                tr_loss_scaling_factor = (t * self.tr_sigma_max + (1 - t) * self.tr_sigma_min) ** 2
            else:
                tr_loss_scaling_factor = 1.0
            
            if self.separate_tr_loss:
                gt_tr_norm = torch.norm(gt_tr, dim=-1, keepdim=True)
                gt_tr_axis = gt_tr / (gt_tr_norm + 1e-6)

                pred_tr_norm = torch.norm(tr_pred, dim=-1, keepdim=True)
                pred_tr_axis = tr_pred / (pred_tr_norm + 1e-6)

                tr_axis_loss = torch.mean((pred_tr_axis - gt_tr_axis)**2)
                if self.scale_pred_by_t:
                    tr_norm_loss = torch.mean((pred_tr_norm / (t + 1e-6) - gt_tr_norm)**2)
                else:
                    tr_norm_loss = torch.mean((pred_tr_norm - gt_tr_norm)**2)

                tr_loss = 0.5 * (tr_axis_loss + tr_norm_loss / tr_loss_scaling_factor)

            else:
                if self.scale_pred_by_t:
                    tr_loss = torch.mean((tr_pred / (t + 1e-6) - gt_tr)**2) / tr_loss_scaling_factor
                else:
                    tr_loss = torch.mean((tr_pred - gt_tr)**2) / tr_loss_scaling_factor
        else:
            tr_loss = torch.tensor(0.0, device=self.device)

        # rotation loss
        if self.perturb_rot:
            if self.separate_rot_loss:
                gt_rot_angle = torch.norm(gt_rot, dim=-1, keepdim=True)
                gt_rot_axis = gt_rot / (gt_rot_angle + 1e-6)

                pred_rot_angle = torch.norm(rot_pred, dim=-1, keepdim=True)
                pred_rot_axis = rot_pred / (pred_rot_angle + 1e-6)

                rot_axis_loss = torch.mean((pred_rot_axis - gt_rot_axis)**2)
                if self.scale_pred_by_t:
                    rot_angle_loss = torch.mean((pred_rot_angle / (t + 1e-6) - gt_rot_angle)**2)
                else:
                    rot_angle_loss = torch.mean((pred_rot_angle - gt_rot_angle)**2)
                rot_loss = 0.5 * (rot_axis_loss + rot_angle_loss)

            else:
                if self.scale_pred_by_t:
                    rot_loss = torch.mean((rot_pred / (t + 1e-6) - gt_rot)**2)
                else:
                    rot_loss = torch.mean((rot_pred - gt_rot)**2)
        else:
            rot_loss = torch.tensor(0.0, device=self.device)
        
        # v loss
        if self.use_v_loss:
            if self.scale_pred_by_t or self.pred_distance:
                v_loss = torch.mean((v - vt_gt)**2)
            else:
                v_loss = torch.mean((v * t - vt_gt)**2)
            if self.scale_tr_loss_by_sigma_max:
                v_loss = v_loss / self.tr_sigma_max
            elif self.scale_tr_loss_by_sigma_t:
                v_loss = v_loss / (t * self.tr_sigma_max + (1 - t) * self.tr_sigma_min)
        else:
            v_loss = torch.tensor(0.0, device=self.device)

        # contrastive loss
        # modified from https://github.com/yilundu/ired_code_release/blob/main/diffusion_lib/denoising_diffusion_pytorch_1d.py
        if self.use_contrastive_loss:
            energy_gt = self.net(batch_gt, return_energy=True)
            energy_stack = torch.stack([energy_gt, energy_noised], dim=-1)
            target = torch.zeros([], device=energy_stack.device)
            el_loss = F.cross_entropy(-1 * energy_stack, target.long(), reduction='none')
        else: 
            el_loss = torch.tensor(0.0, device=self.device)

        bce_logits_loss = nn.BCEWithLogitsLoss()
        # distogram loss
        if self.use_dist_loss:
            gt_dist = torch.norm((batch_gt["rec_pos"][:, None, 1, :] - batch_gt["lig_pos"][None, :, 1, :]), dim=-1, keepdim=True)
            dist_loss = distogram_loss(outputs["dist_logits"], gt_dist)
        else:
            dist_loss = torch.tensor(0.0, device=self.device)

        # interface loss
        if self.use_interface_loss:
            gt_ires = get_interface_residue_tensors(batch_gt["rec_pos"][:, 1, :], batch_gt["lig_pos"][:, 1, :])
            ires_loss = bce_logits_loss(outputs["ires_logits"], gt_ires)
        else:
            ires_loss = torch.tensor(0.0, device=self.device)

        # confidence loss
        if self.use_confidence_loss:
            label = (l_rmsd < 5.0).float()
            conf_loss = bce_logits_loss(outputs["confidence_logits"], label)
        else:
            conf_loss = torch.tensor(0.0, device=self.device)

        # total losses
        loss = tr_loss + rot_loss + v_loss + 0.1 * (ec_loss + el_loss+ conf_loss + dist_loss + ires_loss)
        losses = {
            "tr_loss": tr_loss, 
            "rot_loss": rot_loss, 
            "v_loss": v_loss,
            "ec_loss": ec_loss, 
            "el_loss": el_loss, 
            "dist_loss": dist_loss, 
            "ires_loss": ires_loss,
            "conf_loss": conf_loss,
            "loss": loss,
        }

        return losses

    def modify_coords(self, lig_pos, rot_update, tr_update):
        cen = lig_pos.mean(dim=(0, 1))
        rot = axis_angle_to_matrix(rot_update.squeeze())
        tr = tr_update.squeeze()
        lig_pos = (lig_pos - cen) @ rot.T + cen
        lig_pos = lig_pos + tr
        return lig_pos

    def move_to_lig_center(self, batch):
        center = batch["lig_pos"].mean(dim=(0, 1))
        batch["rec_pos"] = batch["rec_pos"] - center
        batch["lig_pos"] = batch["lig_pos"] - center

    def step(self, batch, batch_idx):
        rec_x = batch['rec_x'].squeeze(0)
        lig_x = batch['lig_x'].squeeze(0)
        rec_pos = batch['rec_pos'].squeeze(0)
        lig_pos = batch['lig_pos'].squeeze(0)
        is_homomer = batch['is_homomer']

        # wrap to a batch
        batch = {
            "rec_x": rec_x,
            "lig_x": lig_x,
            "rec_pos": rec_pos,
            "lig_pos": lig_pos,
            "is_homomer": is_homomer,
        }

        # get losses
        losses = self.loss_fn(batch)
        return losses
    
    def training_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        for loss_name, indiv_loss in losses.items():
            self.log(
                f"train/{loss_name}", 
                indiv_loss, 
                batch_size=1,
            )
        return losses["loss"]

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)
    
    def on_validation_model_train(self, *args, **kwargs):
        super().on_validation_model_train(*args, **kwargs)
        torch.set_grad_enabled(True)

    def validation_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        for loss_name, indiv_loss in losses.items():
            self.log(
                f"val/{loss_name}", 
                indiv_loss, 
                batch_size=1,
            )
        return losses["loss"]

    def test_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        for loss_name, indiv_loss in losses.items():
            self.log(
                f"test/{loss_name}", 
                indiv_loss, 
                batch_size=1,
            )
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        return optimizer

# helper functions

def get_interface_residue_tensors(set1, set2, threshold=8.0):
    device = set1.device
    n1_len = set1.shape[0]
    n2_len = set2.shape[0]
    
    # Calculate the Euclidean distance between each pair of points from the two sets
    dists = torch.cdist(set1, set2)

    # Find the indices where the distance is less than the threshold
    close_points = dists < threshold

    # Create indicator tensors initialized to 0
    indicator_set1 = torch.zeros((n1_len, 1), device=device)
    indicator_set2 = torch.zeros((n2_len, 1), device=device)

    # Set the corresponding indices to 1 where the points are close
    indicator_set1[torch.any(close_points, dim=1)] = 1.0
    indicator_set2[torch.any(close_points, dim=0)] = 1.0

    return torch.cat([indicator_set1, indicator_set2], dim=0)

def get_rmsd(pred, label):
    rmsd = torch.sqrt(torch.mean(torch.sum((pred - label) ** 2.0, dim=-1)))
    return rmsd

#----------------------------------------------------------------------------
# Testing run

@hydra.main(version_base=None, config_path="/scratch4/jgray21/lchu11/graylab_repos/DFMDock/configs/model", config_name="FlowMatchingDock.yaml")
def main(conf: DictConfig):
    dataset = DockingDataset(
        dataset='dips_train',
        training=True,
        use_esm=True,
    )

    subset_indices = [0]
    subset = data.Subset(dataset, subset_indices)

    #load dataset
    dataloader = DataLoader(subset)
    
    model = FlowMatchingDock(
        model=conf.model, 
        experiment=conf.experiment
    )
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=10, inference_mode=False)
    trainer.validate(model, dataloader)

if __name__ == '__main__':
    main()
