import fnmatch
import glob
import os

import numpy as np
import pandas as pd

import project.utils.transforms as local_transforms
from plyfile import PlyData
import pytorch_lightning as pl
import torch
import torch_geometric as ptg
import torch_geometric.transforms as ptg_transforms
from torch.utils.data import random_split
from torch_geometric.data import Data, DataLoader, InMemoryDataset


class CompleteDental3DInMemoryDataset(InMemoryDataset):
    """ Dataset containing dental casts with complete dentition. """

    def __init__(self, root, label_names, pre_process=None, transform=None,
                 pre_transform=None):
        self.label_names, self.pre_process = label_names, pre_process
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(p) for p in self.raw_paths]

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        return glob.glob(os.path.join(self.raw_dir, 'objs/*/*/*/*.ply'))

    def get_path_by_id(self, ids):
        # look for the first obj that is in the StudyID directory
        # assumption: files with same StudyIDs are duplicates

        if isinstance(ids, str): ids = [ids]

        # copy paths for speed
        raw_paths = set(self.raw_paths.copy())

        # look for paths structured as
        # e.g. /objs/Controls/Philippines/PH12281/PH12281MAX.obj
        paths = [
            fnmatch.filter(
                raw_paths,
                os.path.join(self.raw_dir, 'objs/*/*/{}/*.ply'.format(id)))[0]
            for id in ids
        ]

        return paths

    def download(self):
        raise NotImplementedError(
            'Dental is a private, medical dataset that can not be ' \
            'downloaded freely. Please move the data to the {}/raw '\
            'directory'.format(self.root))

    def read_ply(self, path, read_color = False, read_features = False):
        ply_data = PlyData.read(path)

        data_template = Data()

        data_template.pos = torch.from_numpy(np.stack(
            [ply_data['vertex']['x'], ply_data['vertex']['y'],ply_data['vertex']['z']],-1))

        data_template.face = torch.from_numpy(np.stack(
            ply_data['face']['vertex_indices']).T).long()

        if read_features:
            data_template.x = torch.from_numpy(ply_data['vertex']['quality'])[:,None]

        if read_color:
            data_template.mesh_color = torch.from_numpy(np.stack(
                [ply_data['vertex']['red'], ply_data['vertex']['green'],
                ply_data['vertex']['blue']], -1))

        return data_template

    def read_metadata_as_dataframe(self):
        metadata = pd.concat([
            pd.read_excel(f) for f in glob.glob(os.path.join(
                self.raw_dir, 'metadata', '*2020.11.13.xlsx'))])

        # drop rows with duplicate StudyIDs
        metadata = metadata.drop_duplicates(subset=['StudyID'])

        # select participants that do not suffer from orofacial cleft
        metadata = metadata[metadata['Cleft_Type'] == 'Unaffected']

        # 0 total missing other_teeth
        metadata = metadata[metadata['TOT_OMT'] == 0]

        # permanent dentition only
        metadata = metadata[metadata['DentitionType'] == 'Permanent']

        # drop all rows that have a NAN for one of the landmark coordinates
        metadata = metadata.dropna(
            axis=0, how='any', subset=[
                'IP_X', 'IP_Y', 'IP_Z',
                'CR_X', 'CR_Y', 'CR_Z',
                'CL_X', 'CL_Y', 'CL_Z',
                '6R_ER_X', '6R_ER_Y', '6R_ER_Z',
                '6L_EL_X', '6L_EL_Y', '6L_EL_Z',
                'CM_X', 'CM_Y', 'CM_Z',
                'MM_X', 'MM_Y', 'MM_Z'])

        # subjects to be excluded
        excluded_subs = [
            'NG13294', #– Landmarks still don’t align to mesh – possibly the landmarks or the scan were given the incorrect ID at Pittsburgh.
            'CO12032' #– is missing teeth, but this is not correctly recorded in the metadata files.
        ]


        # remove rows that have a StudyID in excluded subs
        metadata = metadata[~metadata['StudyID'].isin(excluded_subs)]

        # ensure determinism
        metadata = metadata.sort_values('StudyID').reset_index(drop=True)

        assert len(metadata) == 1045, 'expected a total of 1045 individuals in' \
            'the simple complete data, but got {} in metadata'.format(len(metadata))

        df = metadata[['StudyID', *self.label_names]]

        assert not df.isnull().values.any(
        ), 'Some of the landmarks are NAN, did you filter correctly?'

        return df

    def process(self):
        # read all the filtered metadata in one dataframe

        data = self.read_metadata_as_dataframe()

        data['path'] = self.get_path_by_id(data['StudyID'].values)

        assert len(data['path']) == 1045, 'expected a total of 1045 individuals in' \
            'the simple complete data, but got {} in objs'.format(len(raw_paths))

        # for debug purposes
        #data = data[:10]

        # read the .ply files of the meshes, format Data(pos, face)
        data_list = []

        # number of landmarks is equal to the number of columns minus the path and StudyID /3
        assert (data.shape[1] - 2) % 3 == 0, "Landmarks are multiple of 3"
        n_landmarks = int((data.shape[1] - 2) / 3)

        # format Data(pos, face, landmark)
        for index, row in data.iterrows():
            data_obj = self.read_ply(row['path'], read_color=True, read_features=True)

            # read the landmarks as floats
            landmarks = row.drop(['path', 'StudyID']).values.astype(np.float32)

            # store landmarks in [-1,3] format
            data_obj['landmark'] = torch.tensor(
                np.split(landmarks, n_landmarks))

            # store the StudyID to be ablt to retrieve extra info during test-phase
            data_obj['identity'] = row['StudyID']
            data_list.append(data_obj)

        if self.pre_process is not None:
            data_list = self.pre_process(data_list, data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # if the pre-transform is invertible, save the parameters so they can
        # be inverted during the testing phase
        if hasattr(self.pre_transform,
                   'invertible') and self.pre_transform.invertible:
            self.pre_transform.store()


class HoldoutDataset(pl.LightningDataModule):
    def __init__(self,
                 root,
                 batch_size=32,
                 labels=None,
                 pre_process=None,
                 transform=None,
                 pre_transform=None,
                 **kwargs):

        self.root = root
        self.pre_process = pre_process
        self.transform = transform
        self.pre_transform = pre_transform
        self.batch_size = batch_size
        self.labels = [
            'IP_X', 'IP_Y', 'IP_Z', 'CR_X', 'CR_Y', 'CR_Z', 'CL_X', 'CL_Y',
            'CL_Z', '6R_ER_X', '6R_ER_Y', '6R_ER_Z', '6L_EL_X', '6L_EL_Y',
            '6L_EL_Z', 'CM_X', 'CM_Y', 'CM_Z', 'MM_X', 'MM_Y', 'MM_Z'
        ] if labels is None else labels

        super().__init__(**kwargs)

    @property
    def n_features(self):
        return 0

    @property
    def label_names(self):
        return self.labels

    @property
    def n_labels(self):
        return len(self.label_names)

    @property
    def metadata_as_df(self):
        return self.dataset.read_metadata_as_dataframe()

    def get_path_by_id(self, ids):
        return self.dataset.get_path_by_id(ids)

    def get_mesh_by_id(self, ids):
        paths = self.dataset.get_path_by_id(ids)

        return [self.dataset.read_ply(p) for p in paths]

    def setup(self, stage=None):
        self.dataset = CompleteDental3DInMemoryDataset(
            self.root,
            self.labels,
            pre_process=self.pre_process,
            pre_transform=self.pre_transform,
            transform=self.transform)

        size = len(self.dataset)
        sz_train, sz_val = round(0.7 * size), round(0.2 * size)
        sz_test = size - (sz_train + sz_val)

        # split the dataset in training validation and test set
        # seeded to ensure replicatibility
        self.data_train, self.data_val, self.data_test = random_split(
            self.dataset, [sz_train, sz_val, sz_test],
            generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.batch_size,
                          num_workers=16,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          batch_size=self.batch_size,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=1, shuffle=False)

    def inference_train_dataloader(self):
        return DataLoader(self.data_train, batch_size=1, shuffle=False)

    def inference_val_dataloader(self):
        return DataLoader(self.data_val, batch_size=1, shuffle=False)

    def inference_sets(self):
        return {
            'train': self.inference_train_dataloader,
            'val': self.inference_val_dataloader,
            'test': self.test_dataloader
        }


class CurvatureCompleteDentalDataModule(HoldoutDataset):
    """
    This model uses the principal directions of the curvature as features
    and uses pca to normalise the landmarks
    """
    def __init__(self, batch_size=32, num_point_samples=4096):
        root = '../data/CurvatureComplete/'

        prep_dir = os.path.join(root, 'prep_parameters')

        # Generate standardised landmarks to setup pca
        training_set_landmarks = torch.tensor(
            pd.read_csv(os.path.join(
                root, 'normalised_train_data_landmarks.csv'
            ), header=0, index_col=0).values[:, 1:].astype('float32')).view(-1, 7, 3)

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

                # normalise landmarks
                local_transforms.PCAProjectLandmarks(
                    training_set_landmarks, n_components=3, invertible=True),

                # generate vertex features
                ptg_transforms.GenerateMeshNormals(),

                # remove labels used in preprocessing
                local_transforms.LabelCloner('pos', 'mesh_vert'),
                local_transforms.LabelCleaner(['edge_index'])
            ],
            invertible=True,
            skip_non_invertible=True,
            store_directory=prep_dir).load_parameters()

        transform = local_transforms.InvertibleCompose(
            [
                local_transforms.LabelCloner('continuous_curvature', 'x'),
                local_transforms.SamplePoints(num_point_samples,
                                              remove_faces=False,
                                              include_normals=True,
                                              include_features=True),
                local_transforms.MergeLabels('norm'),
                local_transforms.ZNormalise('x'),
                local_transforms.LabelCleaner([
                    'mesh_vert', 'norm', 'continuous_curvature', 'face',
                    'mesh_norm', 'discrete_curvature'
                ])
            ],
            skip_non_invertible=True,
            invertible=True)

        super().__init__(root,
                         batch_size=batch_size,
                         transform=transform,
                         pre_transform=pre_transform)

    @property
    def n_features(self):
        return 4


class PatchBasedCompleteDentalDataModule(HoldoutDataset):
    """
    This is the simplest data module possible, it does not perform
    any data augmentation

    """
    def __init__(self,
                 landmark='IP',
                 batch_size=64,
                 num_point_samples=2048,
                 patch_size=8.):
        root = '../data/PatchBasedCompleteDentalDataModule_{}/'.format(
            landmark)

        self.prep_dir = os.path.join(root, 'prep_parameters')
        labels = [landmark + '_X', landmark + '_Y', landmark + '_Z']
        gt_labels = [
            'gt_' + landmark + '_X', 'gt_' + landmark + '_Y',
            'gt_' + landmark + '_Z'
        ]

        # read ids used in training
        training_ids = pd.read_csv(
            os.path.join(
                root, 'normalised_train_data_landmarks.csv'), header=0, index_col=0)['StudyID']

        # read predicted landmarks for training examples
        landmarks = pd.concat([
            pd.read_csv(f, header=0, index_col=0)
            for f in glob.glob(os.path.join(root, 'v0*.csv'))])
        landmarks = landmarks[landmarks['StudyID'].isin(training_ids.values)]

        assert len(landmarks) == len(training_ids)

        # get ground truth and predicted labels
        gt_landmarks = landmarks[gt_labels].astype(float)
        pred_landmarks = landmarks[labels].astype(float)

        # sample weights are proportional to the error, since these are less likely to occur
        self.sample_weights = torch.from_numpy(
            (pred_landmarks - gt_landmarks.values).abs().sum(1).values)

        # setup preprocessing steps, load them if they were executed before
        pre_transform = local_transforms.InvertibleCompose(
            [
                # convert mesh faces to graph edges
                ptg_transforms.FaceToEdge(remove_faces=False),

                # # add edge_attr containing relative euclidean distance
                ptg_transforms.Distance(norm=False, cat=False),
                # extract patch around
                local_transforms.ExtractGeodesicPatch(
                    patch_size, key='patch_center'),

                # b-normalise
                local_transforms.NormalizeScale(invertible=True),

                # save vertex features
                local_transforms.LabelCloner('x', 'continuous_curvature'),
                ptg_transforms.GenerateMeshNormals(),

                # remove labels used in preprocessing
                local_transforms.LabelCloner('pos', 'mesh_vert'),
                local_transforms.LabelCleaner(['edge_index', 'edge_attr'])
            ],
            invertible=True,
            skip_non_invertible=True,
            store_directory=self.prep_dir).load_parameters()

        pre_process = self.set_patch_centers

        transform = ptg_transforms.Compose([
            local_transforms.LabelCloner('continuous_curvature', 'x'),
            local_transforms.SamplePoints(
                num_point_samples, remove_faces=False, include_normals=True,
                include_features=True),
            local_transforms.MergeLabels('norm'),
            local_transforms.ZNormalise('x'),
            local_transforms.LabelCleaner([
                'mesh_vert', 'norm', 'continuous_curvature', 'patch_center',
                'face', 'mesh_norm', 'mesh_color'
            ])
        ])

        super().__init__(
            root, batch_size=batch_size, pre_process=pre_process, transform=transform, labels=labels,
            pre_transform=pre_transform)

    @property
    def n_features(self):
        return 4

    def set_patch_centers(self, data_list, metadata):
        # Read holistic step predictions
        predicted_landmarks = pd.concat([
            pd.read_csv(f, header=0, index_col=0)
            for f in glob.glob(os.path.join(self.root, 'v0*.csv'))
        ])

        # assign holistic step predictions to d.patch_center
        for d in data_list:
            d.patch_center = torch.from_numpy(
                predicted_landmarks[
                    predicted_landmarks['StudyID'] == d.identity][
                        self.labels].values.astype(np.float32)).float()
        return data_list

    def train_dataloader(self):
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            self.sample_weights, len(self.sample_weights))
        return DataLoader(self.data_train,
                          batch_size=self.batch_size,
                          num_workers=16,
                          sampler=sampler)
