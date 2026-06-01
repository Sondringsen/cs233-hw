"""
Part-Aware PC-AE.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
from torch import nn
from ..in_out.utils import AverageMeter
from ..losses.chamfer import chamfer_loss


class PartAwarePointcloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, part_classifier, part_lambda):
        """ Part-aware AE initialization
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        :param part_classifier: nn.Module acting as the second decoding branch that classifies the point part
        labels.
        :param part_lambda: float for weighing the classification loss.
        """
        super(PartAwarePointcloudAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.part_classifier = part_classifier
        self.loss_1 = chamfer_loss
        self.loss_2 = nn.CrossEntropyLoss()
        self.part_lambda = part_lambda

    
    def forward(self, pointclouds):
        """Forward pass of the AE
            :param pointclouds: B x N x 3
        """
        latent = self.encoder(pointclouds)
        reconstructed = self.decoder(latent).view(pointclouds.shape)

        latent_expanded = latent.unsqueeze(1).expand(-1, pointclouds.shape[1], -1)
        point_features = torch.cat([pointclouds, latent_expanded], dim=-1)  # (B, N, 131)
        classifications = self.part_classifier(point_features)  # (B, N, 4)
        return reconstructed, classifications
    
    # def loss(self, pointclouds, reconstructed, true_classifications, classifications):
    #     output_1 = self.loss_1(reconstructed, pointclouds).mean()
    #     B, N = true_classifications.shape
    #     output_2 = self.loss_2(classifications.reshape(B*N, -1), true_classifications.reshape(B*N))
    #     return output_1 + self.part_lambda * output_2


    def train_for_one_epoch(self, loader, optimizer, device='cuda'):
        """ Train the autoencoder for one epoch based on the Chamfer loss.
        :param loader: (train) pointcloud_dataset loader
        :param optimizer: torch.optimizer
        :param device: cuda? cpu?
        :return: tuple of (combined_loss, chamfer_loss, part_loss) averages for the epoch.
        """
        self.train()
        loss_meter = AverageMeter()
        chamfer_meter = AverageMeter()
        part_meter = AverageMeter()
        for batch in loader:
            pointclouds = batch['point_cloud'].to(device)
            true_classifications = batch['part_mask'].to(device)
            reconstructed, classifications = self(pointclouds)
            chamfer = self.loss_1(reconstructed, pointclouds).mean()
            B, N = true_classifications.shape
            part = self.loss_2(classifications.reshape(B*N, -1), true_classifications.reshape(B*N))
            output = chamfer + self.part_lambda * part
            loss_meter.update(output.item(), len(pointclouds))
            chamfer_meter.update(chamfer.item(), len(pointclouds))
            part_meter.update(part.item(), len(pointclouds))
            optimizer.zero_grad()
            output.backward()
            optimizer.step()

        return loss_meter.avg, chamfer_meter.avg, part_meter.avg

    @torch.no_grad()
    def evaluate_losses(self, loader, device='cuda'):
        """ Evaluate both loss components on a loader without updating weights.
        :return: tuple of (combined_loss, chamfer_loss, part_loss) averages.
        """
        self.eval()
        loss_meter = AverageMeter()
        chamfer_meter = AverageMeter()
        part_meter = AverageMeter()
        for batch in loader:
            pointclouds = batch['point_cloud'].to(device)
            true_classifications = batch['part_mask'].to(device)
            reconstructed, classifications = self(pointclouds)
            chamfer = self.loss_1(reconstructed, pointclouds).mean()
            B, N = true_classifications.shape
            part = self.loss_2(classifications.reshape(B*N, -1), true_classifications.reshape(B*N))
            output = chamfer + self.part_lambda * part
            loss_meter.update(output.item(), len(pointclouds))
            chamfer_meter.update(chamfer.item(), len(pointclouds))
            part_meter.update(part.item(), len(pointclouds))
        return loss_meter.avg, chamfer_meter.avg, part_meter.avg

    @torch.no_grad()
    def embed(self, pointclouds):
        """ Extract from the input pointclouds the corresponding latent codes.
        :param pointclouds: B x N x 3
        :return: B x latent-dimension of AE
        """
        # Identical to the non part aware one
        return self.encoder(pointclouds)

    @torch.no_grad()
    def reconstruct(self, loader, device='cuda'):
        """ Reconstruct the point-clouds via the AE.
        :param loader: pointcloud_dataset loader
        :param device: cpu? cuda?
        :return: Left for students to decide
        """
        # Almost identical to the non part aware one
        reconstructed_pointclouds = []
        for batch in loader:
            reconstructed_pointclouds.append(self(batch['point_cloud'].to(device))[0])
        return torch.cat(reconstructed_pointclouds, dim=0) 
