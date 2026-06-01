"""
PC-AE.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
from torch import nn
from ..in_out.utils import AverageMeter
from ..losses.chamfer import chamfer_loss


class PointcloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        """ AE constructor.
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        """
        super(PointcloudAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, pointclouds):
        """Forward pass of the AE
            :param pointclouds: B x N x 3
        """
        latent = self.encoder(pointclouds)
        return self.decoder(latent).view(pointclouds.shape)

        

    def train_for_one_epoch(self, loader, optimizer, device='cuda'):
        """ Train the autoencoder for one epoch based on the Chamfer loss.
        :param loader: (train) pointcloud_dataset loader
        :param optimizer: torch.optimizer
        :param device: cuda? cpu?
        :return: (float), average loss for the epoch.
        """        
        self.train()
        loss_meter = AverageMeter()
        for idx, batch in enumerate(loader):
            batch = batch['point_cloud'].to(device)

            pred = self(batch)
            loss = chamfer_loss(batch, pred).mean()
            loss_meter.update(loss.item(), len(batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss_meter.avg, loss_meter.avg, 0.0

    @torch.no_grad()
    def evaluate_losses(self, loader, device='cuda'):
        """ Evaluate chamfer loss on a loader without updating weights.
        :return: tuple of (combined_loss, chamfer_loss, part_loss) — part_loss is 0 for this model.
        """
        self.eval()
        loss_meter = AverageMeter()
        for batch in loader:
            pointclouds = batch['point_cloud'].to(device)
            pred = self(pointclouds)
            loss = chamfer_loss(pointclouds, pred).mean()
            loss_meter.update(loss.item(), len(pointclouds))
        return loss_meter.avg, loss_meter.avg, 0.0

    @torch.no_grad()
    def embed(self, pointclouds):
        """ Extract from the input pointclouds the corresponding latent codes.
        :param pointclouds: B x N x 3
        :return: B x latent-dimension of AE
        """
        return self.encoder(pointclouds)
        

    @torch.no_grad()
    def reconstruct(self, loader, device='cuda'):
        """ Reconstruct the point-clouds via the AE.
        :param loader: pointcloud_dataset loader
        :param device: cpu? cuda?
        :return: Left for students to decide
        """
        reconstructed_pointclouds = []
        for batch in loader:
            reconstructed_pointclouds.append(self(batch['point_cloud'].to(device)))
        return torch.cat(reconstructed_pointclouds, dim=0)