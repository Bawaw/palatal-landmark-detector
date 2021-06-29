import logging
import math
import os
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors

import dill
import torch
import torch.nn.functional as F
import torch_geometric.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.pool import knn
from torch_geometric.transforms import Center
from torch_geometric.utils import subgraph, add_self_loops
from torch_scatter import scatter

                                ############
                                # GENERICS #
                                ############

class Transform:
    """ Transform base class. """

    def _transform(self):
        raise NotImplementedError('Transform not implemented')

    def __call__(self, data):
        return self._transform(data)


class InvertibleTransform(Transform):
    """ torch_geometric.data.Data(identity) transformation that can also be
    inverted using.invert()
    """

    def __init__(self, invertible=False, store_directory=None, **kwargs):
        self.invertible = invertible
        self.store_directory = store_directory
        super().__init__(**kwargs)

    def _invert(self):
        raise NotImplementedError('Invert not implemented')

    def invert(self, data):
        if not self.invertible:
            raise Exception(
                'Can not call invert if invertible flag was not set in transform initialisation.')

        assert not hasattr(data, 'batch') or data.batch.max(
        ) == 0, 'Batching is not supported for transformation inverse.'

        return self._invert(data)

    def store(self):
        """ Store parameters for persistence."""

        assert self.store_directory is not None, 'Should pass store_directory to {} '\
        'if you want to store it.'.format(type(self).__name__)

        if not os.path.exists(self.store_directory):
            os.makedirs(self.store_directory)

        path = os.path.join(self.store_directory, type(self).__name__ + '.pt')
        torch.save(self, path, pickle_module=dill)

    def load_parameters(self):
        assert self.store_directory is not None, 'Should pass store_directory to {} ' \
        'if you want to load it.'.format(type(self).__name__)

        path = os.path.join(self.store_directory, type(self).__name__ + '.pt')
        if os.path.isfile(path):
            # pickle needs the mocule to be in the same directory
            # https://github.com/pytorch/pytorch/issues/3678
            import project
            sys.path.insert(0, os.path.dirname(project.__file__))

            loaded_invertible_compose = torch.load(path, pickle_module=dill)

            assert repr(loaded_invertible_compose) == repr(self), 'Loading {}, but signature is different.'\
                ' Current: {} and Loaded: {}'.format(type(self).__name__, repr(self),
                                                     repr(loaded_invertible_compose))
            return loaded_invertible_compose
        else:
            logging.warning(
                'Failed to load transform {}, file does not exist {}. Creating new transform...'
                .format(type(self).__name__, path))
            return self


class InvertibleCompose(InvertibleTransform):
    # This code is copy_pasted from TORCH_GEOMETRIC.TRANSFORMS.COMPOSE
    # and slightly adapted to allow for transform inversion
    """Composes several transforms together that can be composed.
    Args:
        transforms (list of :obj:`transform` objects): List of transforms to
            compose.
    """

    def __init__(self, transforms, invertible=False, skip_non_invertible=False, **kwargs):

        self.transforms = transforms
        self.invertible = invertible
        self.skip_non_invertible = skip_non_invertible

        if invertible and not skip_non_invertible:
            assert all([t.invertible for t in transforms]), 'Can\'t setup invertible' \
                'composition if not all transforms are invertible.'

        super().__init__(invertible, **kwargs)

    def _transform(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def _invert(self, data):
        for t in reversed(self.transforms):
            if hasattr(t, 'invertible') and t.invertible:
                data = t.invert(data)

        return data

    def __repr__(self):
        args = ['    {},'.format(t) for t in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))


                                ##############
                                # ESSENTIALS #
                                ##############

class ExtractGeodesicPatch(object):
    """ extract a geodesic patch around node_idx with a radius of max_dist"""

    def __init__(self, max_dist, max_num_hops=1000, key='landmark'):
        self.max_dist, self.max_num_hops, self.key = max_dist, max_num_hops, key

    def _mask_geodesic_patch(self, node_idx, data):
        pos, edge_index, edge_attr = data.pos, data.edge_index, data.edge_attr
        col, row = edge_index
        num_nodes = pos.shape[0]

        node_mask = row.new_empty(num_nodes, dtype=torch.bool)
        edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

        dists = float("inf") * torch.ones([num_nodes], dtype=torch.float)

        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device).flatten()
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]
        dists[node_idx] = 0

        for _ in range(self.max_num_hops):
            node_mask.fill_(False)

            # take the nodes from previous hop
            node_mask[subsets[-1]] = True

            # find the connected nodes
            edge_mask = torch.index_select(node_mask, 0, row)

            # select new connections
            target, source = edge_index[:, edge_mask]

            # distances between new connections
            new_dists = dists[source] + edge_attr[edge_mask].view(-1)

            # concatenate previous dists and new dists for targets
            new_and_old_dists = torch.cat([new_dists, dists[target.unique()]])
            new_and_old_targets = torch.cat([target, target.unique()])

            # pick the smallest distance
            uq_verts, vert_indices = new_and_old_targets.unique(
                return_inverse=True)
            min_dists = scatter(new_and_old_dists, vert_indices, reduce='min')
            dists[uq_verts] = min_dists

            # nodes that are within distance bound
            in_bound_nodes = uq_verts[min_dists < self.max_dist]

            # break if no new nodes are found
            if (subsets[-1].shape == in_bound_nodes.shape) and (
                (subsets[-1] == in_bound_nodes).all()):
                break

            # add within bound nodes to node_mask
            subsets.append(in_bound_nodes)

        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

        node_mask.fill_(False)
        node_mask[subset] = True

        return node_mask

    def __call__(self, data):
        # find idx of landmark
        _, node_idx = knn(data.pos, data[self.key].view(-1, 3), k=1)

        # extract patch around vertex
        mask = self._mask_geodesic_patch(node_idx, data)
        vert_indices = torch.nonzero(mask).view(-1)

        # update datastructure
        data.pos = data.pos[mask]
        data.edge_index, data.edge_attr = subgraph(
            vert_indices, data.edge_index, data.edge_attr, relabel_nodes=True)

        if hasattr(data, 'x'):
            data.x = data.x[mask]

        if hasattr(data, 'face'):
            # only keep faces of which all 3 edges are in the mask
            data.face = data.face[:, (
                data.face[..., None] == vert_indices.view(-1)).any(-1).all(0)]

            # remap faces to match new vertex indices
            index_mapping = torch.zeros(mask.shape, dtype=torch.long)
            index_mapping[mask] = torch.arange(data.pos.shape[0])
            data.face = index_mapping[data.face]

        return data

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.max_dist,
                                       self.max_num_hops, self.key)


class FilterVerts(object):
    """ Remove vertices based on the mask returned by the operator."""

    def __init__(self, operator=None):
        self.operator = operator

    def prune(self, data, node_mask):
        data.pos = data.pos[node_mask]

        if hasattr(data, 'x'):
            data.x = data.x[node_mask]

        if hasattr(data, 'continuous_curvature'):
            data.continuous_curvature = data.continuous_curvature[node_mask]

        if hasattr(data, 'face'):
            vert_indices = torch.nonzero(node_mask)

            # only keep faces of which all 3 edges are in the mask
            data.face = data.face[:, (
                data.face[..., None] == vert_indices.view(-1)).any(-1).all(0)]

            # remap faces to match new vertex indices
            index_mapping = torch.zeros(node_mask.shape, dtype=torch.long)
            index_mapping[node_mask] = torch.arange(data.pos.shape[0])
            data.face = index_mapping[data.face]
        return data

    def __call__(self, data):
        node_mask = self.operator(data)
        return self.prune(data, node_mask)

    def __repr__(self):
        return '{}(operator)'.format(self.__class__.__name__)


class PropageteFeaturesToLocalExtremes(MessagePassing):
    """Perform thresholding based on a propagated neighbourhood average."""

    def __init__(self, c=1, center=0.85):
        super().__init__(aggr="mean")
        self.c, self.center = c, center

    def forward(self, data):
        if data.x.dim() == 1:
            data.x = data.x[:, None]
        data.x = self.propagate(data.edge_index, x=data.x)
        return data

    def message(self, x_i, x_j):
        # the message pushes the node features to extremes
        messages = F.sigmoid(self.c*(x_j-self.center))

        # clamp messages to nodes that have already converged
        messages[
            torch.isclose(x_i, torch.tensor(0.), atol=0.01)] = 0.
        messages[
            torch.isclose(x_i, torch.tensor(1.), atol=0.01)] = 1.
        return messages

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class SamplePoints(object):
    # This code is copy_pasted from TORCH_GEOMETRIC.TRANSFORMS.SAMPLE_POINTS
    # and slightly adapted to allow for feature interpolation
    r"""Uniformly samples :obj:`num` points on the mesh faces according to
    their face area.

    Args:
        num (int): The number of points to sample.
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed. (default: :obj:`True`)
        include_normals (bool, optional): If set to :obj:`True`, then compute
            normals for each sampled point. (default: :obj:`False`)
    """

    def __init__(self, num, remove_faces=True, include_normals=False, include_features=False):
        self.num = num
        self.remove_faces = remove_faces
        self.include_normals = include_normals
        self.include_features = include_features

    def __call__(self, data):
        pos, face = data.pos, data.face
        assert pos.size(1) == 3 and face.size(0) == 3

        pos_max = pos.abs().max()
        pos = pos / pos_max

        area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
        area = area.norm(p=2, dim=1).abs() / 2

        prob = area / area.sum()
        sample = torch.multinomial(prob, self.num, replacement=True)
        face = face[:, sample]

        frac = torch.rand(self.num, 2, device=pos.device)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]

        if self.include_normals:
            data.norm = torch.nn.functional.normalize(vec1.cross(vec2), p=2)

        if self.include_features:
            feature_max = data.x.max()
            x = data.x / feature_max
            feature_vec1 = x[face[1]] - x[face[0]]
            feature_vec2 = x[face[2]] - x[face[0]]

            interpolated_feature = x[face[0]]
            interpolated_feature += frac[:, :1] * feature_vec1
            interpolated_feature += frac[:, 1:] * feature_vec2
            data.x = interpolated_feature * feature_max

        pos_sampled = pos[face[0]]
        pos_sampled += frac[:, :1] * vec1
        pos_sampled += frac[:, 1:] * vec2

        pos_sampled = pos_sampled * pos_max
        data.pos = pos_sampled

        if self.remove_faces:
            data.face = None

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num)


                            #################
                            # NORMALISATION #
                            #################
class ZNormalise:
    def __init__(self, key):
        self.key = key

    def __call__(self, data):
        data[self.key] = (data[self.key] -
                          data[self.key].mean(0)) / data[self.key].std(0)
        return data

class UnitNormaliseScalar(object):
    """Normalise scalar to unit interval."""

    def __init__(self, key = 'x'):
        self.key = key

    def __call__(self, data):
        data[self.key] = (
            data[self.key] - data[self.key].min()) / (data[self.key].max() - data[self.key].min())

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.key)

class NormalizeScale(InvertibleTransform):
    # This code is copy_pasted from TORCH_GEOMETRIC.TRANSFORMS.NORMALIZE_SCALE
    # and slightly adapted to be able to invert the normalisation for
    # both the landmarks and vertices
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`.
    """

    def __init__(self, **kwargs):
        self.scales = {}
        self.means = {}
        super().__init__(**kwargs)

    def _transform(self, data):
        # mean normalise
        data_mean = data.pos.mean(dim=-2, keepdim=True)
        data.pos = data.pos - data_mean

        # normalise scale
        scale = (1 / data.pos.abs().max()) * 0.999999

        data.pos = data.pos * scale

        if self.invertible:
            # if the object is batched the identity will be a collection
            id = data.identity[0] if hasattr(data, 'batch') else data.identity

            if id in self.means:
                logging.warning(
                    'Calling transform on object that was transformed once, '\
                    'this might result in inconsistent parameters in param.pt.')

            self.means[id] = data_mean
            self.scales[id] = scale

        if hasattr(data, 'landmark'):
            data.landmark = data.landmark - data_mean
            data.landmark = data.landmark * scale

        if hasattr(data, 'patch_center'):
            data.patch_center = data.patch_center - data_mean
            data.patch_center = data.patch_center * scale

        if hasattr(data, 'mesh_vert'):
            data.mesh_vert = data.mesh_vert - data_mean
            data.mesh_vert = data.mesh_vert * scale

        return data

    def _invert(self, data):
        # if the object is batched the identity will be a collection
        id = data.identity[0] if hasattr(data, 'batch') else data.identity

        # load transform params
        data_mean = self.means[id].to(data.pos.device)
        scale = self.scales[id].to(data.pos.device)

        # undo normalise
        if hasattr(data, 'pos'):
            data.pos = data.pos / scale
            data.pos = data.pos + data_mean

        if hasattr(data, 'landmark'):
            data.landmark = data.landmark / scale
            data.landmark = data.landmark + data_mean

        if hasattr(data, 'pred'):
            data.pred = data.pred / scale
            data.pred = data.pred + data_mean

        if hasattr(data, 'patch_center'):
            data.patch_center = data.patch_center / scale
            data.patch_center = data.patch_center + data_mean

        if hasattr(data, 'mesh_vert'):
            data.mesh_vert = data.mesh_vert / scale
            data.mesh_vert = data.mesh_vert + data_mean

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class PCAProjectLandmarks(InvertibleTransform):
    """ Projects components to pca space and normalise the coeffcients based on variance."""

    def __init__(self, data, n_components, **kwargs):
        self.n_components = n_components

        # shape(data): 7xNx3
        # We perform pca for each landmark individually
        data = data.transpose(0, 1)
        self.mean = data.mean(1).unsqueeze(1)
        std_data = data - self.mean

        u, s, v = torch.svd(std_data, some=True)
        self.v, self.s = v[..., :n_components], s[:, :n_components]
        self.std = (std_data @ self.v).std(1)
        super().__init__(**kwargs)

    def _transform(self, data):
        if hasattr(data, 'landmark'):
            data.landmark = (
                (data.landmark[:, None] - self.mean) @ self.v).squeeze() / self.std
        if hasattr(data, 'pred'):
            data.pred = (
                (data.pred[:, None] - self.mean) @ self.v).squeeze() / self.std

        return data

    def _invert(self, data):
        if hasattr(data, 'landmark'):
            data.landmark = (
                ((data.landmark * self.std.to(data.landmark.device))[:, None]
                 @ self.v.to(data.landmark.device).transpose(1, 2)) +
                self.mean.to(data.landmark.device)).squeeze()
        if hasattr(data, 'pred'):
            data.pred = (((data.pred * self.std.to(data.pred.device))[:, None]
                          @ self.v.to(data.pred.device).transpose(1, 2)) +
                         self.mean.to(data.pred.device)).squeeze()

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.n_components)


                            #######################
                            # DATA OBJECT CONTROL #
                            #######################

class MergeLabels(object):
    """Concatenate two labels in one. """

    def __init__(self, from_key, to_key='x', standardise=False):
        self.from_key, self.to_key = from_key, to_key

    def __call__(self, data):
        data[self.to_key] = torch.cat(
            [data[self.to_key], data[self.from_key]], 1)
        return data


class LabelCleaner(object):
    """ Removes the class arg labels from data object """

    def __init__(self, labels):
        self.labels = labels

    def __call__(self, data):
        for label in self.labels:
            if hasattr(data, label):
                data[label] = None
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.labels)


class LabelCloner(object):
    """Clone the label of a data object. """

    def __init__(self, source_key, target_key):
        self.source_key, self.target_key = source_key, target_key

    def __call__(self, data):
        data[self.target_key] = data[self.source_key].clone()
        return data

    def __repr__(self):
        return '{}(source={}, target={})'.format(
            self.__class__.__name__, self.source_key, self.target_key)
