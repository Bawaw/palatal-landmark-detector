import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (PointConv, TopKPooling, fps, global_max_pool,
                                graclus, knn_graph, max_pool, radius)


class DentalNetModule(pl.LightningModule):
    class PointNetBlock(nn.Module):
        def __init__(self,
                     in_channels,
                     int_channels,
                     out_channels,
                     ratio=0.5,
                     r=0.2):
            """
            PointNet++ building block,

            input args:
                in_channels shape: [-1, in_channels + num_dimensions]
                out_channels shape: [-1, out_channels]
            """
            super().__init__()

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.ratio = ratio
            self.r = r

            self.conv = PointConv(
                nn.Sequential(
                    nn.Sequential(nn.Linear(in_channels, int_channels),
                                  nn.BatchNorm1d(int_channels),
                                  nn.ReLU(inplace=True)),
                    nn.Sequential(nn.Linear(int_channels, int_channels),
                                  nn.BatchNorm1d(int_channels),
                                  nn.ReLU(inplace=True)),
                    nn.Sequential(nn.Linear(int_channels, out_channels),
                                  nn.BatchNorm1d(out_channels),
                                  nn.ReLU(inplace=True))))

        def forward(self, x, pos, batch):
            # Sampling Layer
            idx = fps(pos, batch, ratio=self.ratio)
            # Grouping Layer
            row, col = radius(pos,
                              pos[idx],
                              self.r,
                              batch,
                              batch[idx],
                              max_num_neighbors=64)
            edge_index = torch.stack([col, row], dim=0)

            # PointNet Layer
            x = self.conv(x, (pos, pos[idx]), edge_index)
            pos, batch = pos[idx], batch[idx]
            return x, pos, batch

    class MaxPoolBlock(nn.Module):
        def __init__(self, in_channels, int_channels1, int_channels2,
                     out_channels):
            super().__init__()

            self.nn = nn.Sequential(
                nn.Sequential(nn.Linear(in_channels, int_channels1),
                              nn.BatchNorm1d(int_channels1),
                              nn.ReLU(inplace=True)),
                nn.Sequential(nn.Linear(int_channels1, int_channels2),
                              nn.BatchNorm1d(int_channels2),
                              nn.ReLU(inplace=True)),
                nn.Sequential(nn.Linear(int_channels2, out_channels),
                              nn.BatchNorm1d(out_channels),
                              nn.ReLU(inplace=True)))

        def forward(self, x, pos, batch):
            x = self.nn(torch.cat([x, pos], dim=1))
            x = global_max_pool(x, batch)
            return x

    class FullyConnectedBlock(nn.Module):
        def __init__(self, in_channels, int_channels1, int_channels2,
                     out_channels):
            super().__init__()

            self.nn = nn.Sequential(
                # B*1024
                nn.Linear(in_channels, int_channels1),
                nn.BatchNorm1d(int_channels1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                # B*1024
                nn.Linear(int_channels1, int_channels2),
                nn.BatchNorm1d(int_channels2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                # B*512
                nn.Linear(int_channels2, out_channels),
            )

        def forward(self, x):
            return self.nn(x)

    def __init__(self, dataset=None):
        super().__init__()

        print('Initialising network with, feature dim: {}'.format(
            dataset.n_features))

        self.conv1 = DentalNetModule.PointNetBlock(dataset.n_features + 3,
                                                   64,
                                                   128,
                                                   ratio=0.5,
                                                   r=0.015)
        self.conv2 = DentalNetModule.PointNetBlock(128 + 3,
                                                   128,
                                                   256,
                                                   ratio=0.5,
                                                   r=0.04)
        self.max_pool = DentalNetModule.MaxPoolBlock(256 + 3, 256, 512, 1024)
        self.fc = DentalNetModule.FullyConnectedBlock(1024, 1024, 512,
                                                      dataset.n_labels)
        self.dataset = dataset

    def forward(self, data):
        x = data.x if hasattr(data, 'x') else None

        pos, batch = data.pos, data.batch
        x, pos, batch = self.conv1(x, pos, batch)
        x, pos, batch = self.conv2(x, pos, batch)
        x = self.max_pool(x, pos, batch)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        # predict landmarks using this model
        landmark_predictions = self(batch)

        # mse loss between predicted landmarks and actual landmarks
        loss = F.mse_loss(landmark_predictions.view(-1),
                          batch.landmark.view(-1))

        # Logging to TensorBoard by default
        self.log('train_loss', 100 * loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # predict landmarks using this model
        landmark_predictions = self(batch)

        loss = F.mse_loss(landmark_predictions.view(-1),
                          batch.landmark.view(-1))

        # Logging to TensorBoard by default
        self.log('val_loss', 100 * loss)
        return loss

    def test_step(self, batch, batch_idx):
        # predict landmarks using this model
        batch.pred = self(batch).view(batch.landmark.shape)

        # invert the augmentation steps
        if hasattr(self.dataset.transform, "invert"):
            batch = self.dataset.transform.invert(batch)

        # invert the preprocessing steps
        if hasattr(self.dataset.pre_transform, "invert"):
            batch = self.dataset.pre_transform.invert(batch)

        # unpack predictions and ground truth in corresponding headers
        gt_label_names = ['gt_' + lmn for lmn in self.dataset.label_names]

        output = {
            'StudyID': batch.identity[0],
            **dict(zip(self.dataset.label_names,
                       batch.pred.view(-1).tolist())),
            **dict(zip(gt_label_names,
                       batch.landmark.view(-1).tolist()))
        }

        return output

    def write_outputs_in_logdir(self, outputs, file_name, log_dir=None):
        import pandas as pd
        df = pd.DataFrame(outputs)

        if log_dir is not None:
            df.to_csv(os.path.join(log_dir, file_name))

        elif self.logger is not None:
            df.to_csv(os.path.join(self.logger.log_dir, file_name))
        else:
            raise AttributeError(
                'No logger, did you set logger and checkpoint_callback?')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return {
            'optimizer':
            optimizer,
            'lr_scheduler':
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       min_lr=1.e-4,
                                                       factor=0.5,
                                                       verbose=True,
                                                       patience=20),
            'monitor':
            'train_loss'
        }
