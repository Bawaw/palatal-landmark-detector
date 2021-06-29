import os
import torch
import pandas as pd
import project.utils.transforms as local_transforms
import torch_geometric.transforms as ptg_transforms

from torch.utils.data import random_split
from project.dental_data import HoldoutDataset, CompleteDental3DInMemoryDataset

# This script should be executed before all else to generate the normalised training set landmarks

class CurvatureCompleteDentalDataNormalisedLandmarkGenerator(HoldoutDataset):
    def __init__(self, batch_size=32, num_point_samples=4096):
        """ This code runs the preprocessing pipeline ones and stores the standardised
        coordinates. It seems overkill to run the entire pipeline just to normalise the landmarks
        but it is required to find the bounding box. """

        root = '../data/CurvatureComplete_normalisation_only/'

        prep_dir = os.path.join(root, 'prep_parameters')

        # only perform preprocessing upto b-norm of the landmarks
        pre_transform = local_transforms.InvertibleCompose(
            [
                # convert mesh faces to graph edges
                ptg_transforms.FaceToEdge(remove_faces=False),
                local_transforms.LabelCloner('x', 'continuous_curvature'),

                # standardise x so that x ∈ [0,1]
                local_transforms.UnitNormaliseScalar(),

                # propagate curvature feature to local extreme
                local_transforms.PropageteFeaturesToLocalExtremes(c=80),

                # remove vertices based on curvature
                local_transforms.FilterVerts(
                    operator=lambda data: data.x[:,0] < 0.85
                ),

                # b-normalise
                local_transforms.NormalizeScale(invertible=True),
                local_transforms.LabelCloner('x', 'discrete_curvature'),
            ],
            invertible=True,
            skip_non_invertible=True,
            store_directory=prep_dir).load_parameters()


        super().__init__(root,
                         batch_size=batch_size,
                         transform=None,
                         pre_transform=pre_transform)

    def store_normalised_landmarks(self):
        # preprocess the data
        self.dataset = CompleteDental3DInMemoryDataset(
            self.root,
            self.labels,
            pre_process=self.pre_process,
            pre_transform=self.pre_transform,
            transform=self.transform)

        # split the dataset in training validation and test set
        size = len(self.dataset)
        sz_train, sz_val = round(0.7 * size), round(0.2 * size)
        sz_test = size - (sz_train + sz_val)

        # seeded to ensure replicatibility
        self.data_train, self.data_val, self.data_test = random_split(
            self.dataset, [sz_train, sz_val, sz_test],
            generator=torch.Generator().manual_seed(42))

        # for data_obj in self.dataset:
        #     data_obj.mesh_vert = data_obj.pos
        #     from vis.plotter import plot_landmarks_and_pred, plot_data_obj
        #     plot_data_obj(data_obj)


        # the data for the pca normalisation is only taken from the trainingset
        train_data_landmarks = [(d.identity, *d.landmark.view(-1).tolist())
                                for d in self.data_train]
        pd.DataFrame(
            train_data_landmarks, columns=[
                'StudyID', *self.dataset.label_names
            ]).to_csv(os.path.join(self.root, 'normalised_train_data_landmarks.csv'))

if __name__ == '__main__':
    # preprocess data
    num_samples = 4096
    dataset = CurvatureCompleteDentalDataNormalisedLandmarkGenerator(
        batch_size=64, num_point_samples=num_samples)

    # store preprocessed data
    dataset.store_normalised_landmarks()
