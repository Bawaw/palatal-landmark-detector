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


class Transform:
    def __init__(self, inference=False, skip_during_inference=False):
        self.skip_during_inference = skip_during_inference

    def inference_state(self, state=False):
        self.inference = state

    def _transform(self):
        raise NotImplementedError('Transform not implemented')

    def __call__(self, data):
        if self.skip_during_inference and self.inference:
            return data
        else:
            return self._transform(data)


class InvertibleTransform(Transform):
    """
    torch_geometric.data.Data(identity) transformation that can also be
    inverted using.invert()
    """
    def __init__(self, invertible=False, store_directory=None, **kwargs):
        self.invertible = invertible
        self.store_directory = store_directory
        super().__init__(**kwargs)

    @property
    def _invert(self):
        raise NotImplementedError('Invert not implemented')

    def invert(self, data):
        if not self.invertible:
            raise Exception(
                'Can not call invert if invertible flag was not set in transform initialisation.'
            )

        assert not hasattr(data, 'batch') or data.batch.max(
        ) == 0, 'Batching is not supported for transformation inverse.'

        return self._invert(data)

    def store(self):
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
    def __init__(self,
                 transforms,
                 invertible=False,
                 skip_non_invertible=False,
                 **kwargs):
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

    def inference_state(self, state=False):
        for t in self.transforms:
            if isinstance(t, Transform):
                t.inference_state(state)
        super().inference_state(state)

    def _invert(self, data):
        for t in reversed(self.transforms):
            if hasattr(t, 'invertible') and t.invertible:
                data = t.invert(data)
        return data

    def __repr__(self):
        args = ['    {},'.format(t) for t in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))


class NormalizeRotation(InvertibleTransform):
    # This code is copy_pasted from TORCH_GEOMETRIC.TRANSFORMS.NORMALIZE_ROTATION
    # and slightly adapted to be able to invert the normalisation for
    # both the landmarks and vertices
    r"""Rotates all points according to the eigenvectors of the point cloud.
    If the data additionally holds normals saved in :obj:`data.normal`, these
    will be rotated accordingly.
    Args:
        max_points (int, optional): If set to a value greater than :obj:`0`,
            only a random number of :obj:`max_points` points are sampled and
            used to compute eigenvectors. (default: :obj:`-1`)
        sort (bool, optional): If set to :obj:`True`, will sort eigenvectors
            according to their eigenvalues. (default: :obj:`False`)
    """
    def __init__(self, max_points: int = -1, **kwargs):

        self.max_points = max_points
        self.rotation_vectors = {}

        super().__init__(**kwargs)

    def _transform(self, data):

        pos = data.pos

        if self.max_points > 0 and pos.size(0) > self.max_points:
            perm = torch.randperm(pos.size(0))
            pos = pos[perm[:self.max_points]]

        pos = pos - pos.mean(dim=0, keepdim=True)
        C = torch.matmul(pos.t(), pos)
        e, v = torch.eig(C, eigenvectors=True)  # v[:,j] is j-th eigenvector

        indices = e[:, 0].argsort(descending=True)
        v = v.t()[indices].t()

        data.pos = torch.matmul(data.pos, v)

        if hasattr(data, 'landmark'):
            data.landmark = torch.matmul(data.landmark, v)

        # if the object is batched the identity will be a collection
        if self.invertible:
            id = data.identity[0] if hasattr(data, 'batch') else data.identity

            if id in self.rotation_vectors:
                logging.warning(
                    'Calling transform on object that was transformed once, '\
                    'this might result in inconsistent parameters in param.pt.')

            self.rotation_vectors[id] = v

        if 'normal' in data:
            data.normal = F.normalize(torch.matmul(data.normal, v))

        return data

    def _invert(self, data):
        # if the object is batched the identity will be a collection
        id = data.identity[0] if hasattr(data, 'batch') else data.identity

        v_inv = torch.inverse(self.rotation_vectors[id].to(data.pos.device))

        if hasattr(data, 'pos'):
            data.pos = torch.matmul(data.pos, v_inv)

        if hasattr(data, 'mesh_vert'):
            data.mesh_vert = torch.matmul(data.mesh_vert, v_inv)

        if hasattr(data, 'landmark'):
            data.landmark = torch.matmul(data.landmark, v_inv)

        if hasattr(data, 'pred'):
            data.pred = torch.matmul(data.pred, v_inv)

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.max_points)


class LabelCloner(object):
    def __init__(self, source_key, target_key):
        self.source_key, self.target_key = source_key, target_key

    def __call__(self, data):
        data[self.target_key] = data[self.source_key].clone()
        return data

    def __repr__(self):
        return '{}(source={}, target={})'.format(self.__class__.__name__,
                                                 self.source_key,
                                                 self.target_key)


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
        scale = 1 / (self.scales[id] / 0.999999).to(data.pos.device)

        # undo normalise
        if hasattr(data, 'pos'):
            data.pos = data.pos * scale
            data.pos = data.pos + data_mean

        if hasattr(data, 'landmark'):
            data.landmark = data.landmark * scale
            data.landmark = data.landmark + data_mean

        if hasattr(data, 'pred'):
            data.pred = data.pred * scale
            data.pred = data.pred + data_mean

        if hasattr(data, 'patch_center'):
            data.patch_center = data.patch_center * scale
            data.patch_center = data.patch_center + data_mean

        if hasattr(data, 'mesh_vert'):
            data.mesh_vert = data.mesh_vert * scale
            data.mesh_vert = data.mesh_vert + data_mean

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ExtractGeodesicPatch(object):
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
        data.edge_index, data.edge_attr = subgraph(vert_indices,
                                                   data.edge_index,
                                                   data.edge_attr,
                                                   relabel_nodes=True)
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


class RemoveUnconnectedPieces(MessagePassing):
    def __init__(self, convergence_delay=10):
        super().__init__(aggr="max")
        self.convergence_delay = convergence_delay
        self.mesh_pruner = FilterVerts()

    def forward(self, data):
        num_verts = data.pos.shape[0]

        # add an edge from source to source
        edge_index, _ = add_self_loops(data.edge_index,
                                       num_nodes=num_verts)

        # every node starts at its own index
        cluster_index = torch.arange(num_verts).unsqueeze(-1)

        # iterations since last change
        iters_since_change = 0
        for i in range(num_verts):
            # take the maximum of neighbours
            new_cluster_index = self.propagate(edge_index, x=cluster_index)

            # if no change
            if (new_cluster_index == cluster_index).all():
                # assume convergence
                if iters_since_change < self.convergence_delay:
                    break
                else:
                    iters_since_change += 1

            cluster_index = new_cluster_index

        # list unique clusters and count verts in each
        unique_clusters, cluster_counts = cluster_index.unique(
            return_counts=True)

        # take cluster with most verts
        most_common_cluster = unique_clusters[cluster_counts.argmax()]

        # remove vertices in other clusters
        return self.mesh_pruner.prune(
            data, (cluster_index == most_common_cluster).view(-1))

    def message(self, x_i, x_j):
        return x_j

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.convergence_delay)


class NormalPerturbation(Transform):
    def __init__(self, mean=0., std=1., key='pos', **kwargs):
        self.mean, self.std, self.key = mean, std, key
        super().__init__(**kwargs)

    def _transform(self, data):
        perturbation = self.mean + self.std * torch.randn(data[self.key].shape)
        data[self.key] += perturbation
        return data

    def __repr__(self):
        return '{}(mean={}, std={}, key={})'.format(self.__class__.__name__,
                                                    self.mean, self.std,
                                                    self.key)


class PropageteFeaturesToLocalExtremes(MessagePassing):
    def __init__(self, n_iters, c=1, center=0.85):
        super().__init__(aggr="mean")
        self.n_iters = n_iters
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
        return '{}(n_iters={})'.format(self.__class__.__name__, self.n_iters)


class LabelCleaner(object):
    """
    Removes the class arg labels from data object
    """
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
    def __init__(self, source_key, target_key):
        self.source_key, self.target_key = source_key, target_key

    def __call__(self, data):
        data[self.target_key] = data[self.source_key].clone()
        return data

    def __repr__(self):
        return '{}(source={}, target={})'.format(self.__class__.__name__,
                                                 self.source_key,
                                                 self.target_key)


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
    def __init__(self,
                 num,
                 remove_faces=True,
                 include_normals=False,
                 include_features=False):
        self.num = num
        self.remove_faces = remove_faces
        self.include_normals = include_normals
        self.include_features = include_features

    def __call__(self, data):
        pos, face = data.pos, data.face
        assert pos.size(1) == 3 and face.size(0) == 3

        pos_max = pos.max()
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


class ShotDescriptor:
    class QuadrilinearInterpolator:
        def __init__(self, n_bins: list, border_correction: list = None):
            self.n_bins = n_bins
            if border_correction is None:
                self.border_correction = [
                    self._correct_border_bin_cyclic,
                    self._correct_border_bin_rollback,
                    self._correct_border_bin_ignore_outer,
                    self._correct_border_bin_rollback
                ]

            else:
                self.border_correction = border_correction

        def _correct_border_bin_cyclic(self, x, neigh_bin, n_bins):
            bin_width = 1. / n_bins
            neigh_cvalue = ((neigh_bin + 0.5) * bin_width)

            # 1-d is added to bin on other extreme
            neigh_bin = neigh_bin % n_bins
            di_neigh = torch.abs(x - neigh_cvalue) / bin_width
            return di_neigh, neigh_bin, neigh_cvalue

        def _correct_border_bin_rollback(self, x, neigh_bin, n_bins):
            bin_width = 1. / n_bins
            neigh_cvalue = ((neigh_bin + 0.5) * bin_width)

            # 1- d is added to current bin
            neigh_bin = torch.clamp(neigh_bin, 0, n_bins - 1)
            di_neigh = torch.abs(x - neigh_cvalue) / bin_width
            return di_neigh, neigh_bin, neigh_cvalue

        def _correct_border_bin_ignore_outer(self, x, neigh_bin, n_bins):
            invalid_mask = neigh_bin == n_bins

            di_neigh, neigh_bin, neigh_cvalue = self._correct_border_bin_rollback(
                x, neigh_bin, n_bins)

            # ignore the ones on the outer rim
            di_neigh[invalid_mask] = 1
            return di_neigh, neigh_bin, neigh_cvalue

        def interpolate(self, data: list):
            n_q = data[0].shape[0]
            n_dims = 4
            descriptor = torch.zeros(n_q, *self.n_bins)

            bins = [
                (0.99999 * torch.clamp(data[i], 0, 1) * self.n_bins[i]).long()
                for i in range(n_dims)
            ]
            bins.insert(0, torch.arange(n_q))

            for i in range(n_dims):
                x = data[i]

                # distance to bin central value, di ∈ [0, 0.5]
                bin_width = 1. / self.n_bins[i]
                bin_cvalue = ((bins[i + 1] + 0.5) * bin_width)
                di = torch.abs(x - bin_cvalue) / bin_width

                assert di.min() >= 0 and di.max() < 0.51

                # distance to neighbour bin central value, di_neigh ∈ [0, 0.5]
                neigh_bin = bins[i + 1] + (2 * (x > bin_cvalue)) - 1
                di_neigh, neigh_bin, neigh_cvalue = self.border_correction[i](
                    x, neigh_bin, self.n_bins[i])

                assert di_neigh.min() > 0.49 and di_neigh.max() < 1.1, i
                assert i == 2 or torch.allclose(di + di_neigh,
                                                torch.tensor(1.))

                # accumulate in local histograms
                assert (neigh_bin != bins[i + 1]).any(), i
                assert (bins[i + 1].max() <
                        self.n_bins[i]) and (bins[i + 1].min() >= 0)
                _bins = bins.copy()
                _bins[i + 1] = torch.stack([bins[i + 1], neigh_bin])
                descriptor[_bins] = 1 - torch.stack([di, di_neigh])

            return descriptor

    def __init__(self, radius, shape_bins=11, k=63):
        self.radius = radius
        self.shape_bins = shape_bins
        self.k = k
        self.q_interpolator = ShotDescriptor.QuadrilinearInterpolator(
            [8, 2, 2, int(shape_bins)])

    def _disambiguate_axis(self, q_min_p, x, p, q):
        # check angle between connection and local eigenvector
        Sx_plus = 1. * (((q_min_p) * x[p]).sum(-1) >= 0)
        Sx_min = 1. * (((q_min_p) * -x[p]).sum(-1) > 0)

        # aggregate count per feature point
        # shape(Sx_.): = n_p
        Sx_plus = scatter(Sx_plus, p, dim=0, reduce="sum")
        Sx_min = scatter(Sx_min, p, dim=0, reduce="sum")

        assert ((Sx_plus + Sx_min) == p.bincount()).all()

        # swap the direction of the eigenvector for which it is more often not the case
        x[~(Sx_plus >= Sx_min)] = -x[~(Sx_plus >= Sx_min)]

        # if there is ambiguity
        if (Sx_plus == Sx_min).sum() > 0:
            # connections for which the direction is ambiguous
            ambiguous_mask = Sx_plus[p] == Sx_min[p]
            p_amb = p[ambiguous_mask]
            p_amb_indices, p_amb_batch = p_amb.unique(return_inverse=True)

            # dist from the feature points
            q_min_p_amb = q_min_p[ambiguous_mask]
            dist_amb = torch.norm(q_min_p_amb, 2, dim=1)

            # WARNING: we used the mean instead of the median
            p_mean_dist = scatter(dist_amb, p_amb_batch, dim=0, reduce="mean")

            # subset for which distance from the feature points
            # is k closest to the mean distance
            source_index, knn_q = nn.knn(dist_amb, p_mean_dist, self.k,
                                         p_amb_batch,
                                         torch.arange(p_amb_indices.shape[0]))

            knn_p = p_amb_indices[source_index]

            # mean_index to feature_point index

            S_tilde_x_plus = 1 * ((q_min_p_amb[knn_q] * x[knn_p]).sum(-1) >= 0)
            S_tilde_x_min = 1 * ((q_min_p_amb[knn_q] * -x[knn_p]).sum(-1) > 0)

            S_tilde_x_plus = scatter(S_tilde_x_plus,
                                     source_index,
                                     dim=0,
                                     reduce="sum")
            S_tilde_x_min = scatter(S_tilde_x_min,
                                    source_index,
                                    dim=0,
                                    reduce="sum")

            assert ((S_tilde_x_plus +
                     S_tilde_x_min) == source_index.bincount()).all()
            amb_ind = p_amb_indices[S_tilde_x_plus < S_tilde_x_min]
            x[amb_ind] = -x[amb_ind]

        return x

    def __call__(self, data):
        feature_pos = data.pos
        support_pos = torch.cat([data.mesh_vert, data.pos])
        support_norm = torch.cat([data.mesh_norm, data.norm])

        # p: feature point index, q: index of points in support
        # p[i] o--o q[i] represents a source - target connections
        p, q = nn.radius(support_pos,
                         feature_pos,
                         r=self.radius,
                         max_num_neighbors=1000000)

        n_q = q.shape[0]
        q_min_p = support_pos[q] - feature_pos[p]
        d = torch.norm(q_min_p, 2, dim=1)

        # Kind of Covariance matrix M of points in support
        # shape(M): (n_p, 3, 3)
        M = (self.radius - d)[:, None, None] * ((q_min_p).unsqueeze(-1) *
                                                (q_min_p).unsqueeze(-2))

        M = scatter(M, p, dim=0, reduce="sum") / scatter(
            (self.radius - d), p, dim=0, reduce="sum")[:, None, None]

        # eigenvalue decomposition (columns of v are eigenvectors)
        lmbda, v = torch.symeig(M, eigenvectors=True)
        lmbda, v = torch.flip(lmbda, [1]), torch.flip(v, [2])
        assert torch.allclose((torch.matmul(
            torch.matmul(v[0], torch.diag(lmbda[0])), torch.t(v[0]))),
                              M[0],
                              atol=1.e-3)

        # disambiguate the local coordinate system
        v[..., 0] = self._disambiguate_axis(q_min_p, v[..., 0], p, q)
        v[..., 2] = self._disambiguate_axis(q_min_p, v[..., 2], p, q)
        v[..., 1] = torch.cross(v[..., 2], v[..., 0])
        data.v = v

        # pseudo coordinates to local coordinate system
        local_coords = (q_min_p.unsqueeze(1) @ v[p]).squeeze()
        qx, qy, qz = local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]

        assert torch.allclose(
            (local_coords.unsqueeze(1) @ v[p].transpose(-2, -1)).squeeze(),
            q_min_p,
            atol=1.e-3)

        # normalise all data in ω ∈ [0,1[ range
        # azimuth: ω ∈ [-π, π]
        omega = ((math.pi + torch.atan2(qy, qx)) / (2 * math.pi))

        # elevation: ϕ ∈ [0, π]
        phi = (torch.acos(qz / torch.norm(local_coords, 2)) / math.pi)

        # radius: d ∈ [0, self.radius]
        radius = (d / self.radius)

        # angle between eigenvector z and normals: θ ∈ [-1, 1]
        theta = ((1. + (support_norm[q] * v[p, 2]).sum(-1)) / 2)

        # interpolate weights
        descriptor = self.q_interpolator.interpolate(
            [omega, phi, radius, theta])

        # accumulate q in histograms
        descriptor = scatter(descriptor.view(n_q, -1), p, dim=0, reduce="sum")

        # normalise descriptor
        data.x = descriptor / torch.norm(descriptor, 2, dim=1)[:, None]

        return data


class MergeLabels(object):
    def __init__(self, from_key, to_key='x', standardise=False):
        self.from_key, self.to_key = from_key, to_key

    def __call__(self, data):
        data[self.to_key] = torch.cat([data[self.to_key], data[self.from_key]],
                                      1)
        return data


class ZNormalise:
    def __init__(self, key):
        self.key = key

    def __call__(self, data):
        data[self.key] = (data[self.key] -
                          data[self.key].mean(0)) / data[self.key].std(0)
        return data


class PCAProjectLandmarks(InvertibleTransform):
    """
    Projects components in a lower dimensional space
    """
    def __init__(self, data, n_components, **kwargs):
        self.n_components = n_components

        # shape(data): 7xNx3
        data = data.transpose(0, 1)
        self.mean = data.mean(1).unsqueeze(1)
        std_data = data - self.mean

        u, s, v = torch.svd(std_data, some=True)
        self.v, self.s = v[..., :n_components], s[:, :n_components]
        self.std = (std_data @ self.v).std(1)
        super().__init__(**kwargs)

    def _transform(self, data):
        if hasattr(data, 'landmark'):
            data.landmark = ((data.landmark[:, None] - self.mean)
                             @ self.v).squeeze() / self.std
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


class ProjectLabelToSurface:
    def __init__(self,
                 from_label='patch_center',
                 to_label='patch_center',
                 vertex_label='pos'):
        self.from_label = from_label
        self.to_label = to_label
        self.vertex_label = vertex_label

    def __call__(self, data):
        # read labels as numpy array
        surface_points = data[self.vertex_label].numpy()
        points = data[self.from_label].numpy()

        # project numpy array on vertex points
        updated_points = self.project_to(points, surface_points)

        # numpy to tensor and assign to data obj
        data[self.to_label] = torch.from_numpy(updated_points).float()
        return data

    def project_to(self, points, surfacePoints):
        """
        INPUTS:
            points - an n x 3 array of points to project onto the surface
            surfacePoints - a q x 3 array of points defining the surface
        OUTPUTE:
            OUT - an n x 3 array of points projected onto the surface
        """

        nbrs = NearestNeighbors(n_neighbors=3).fit(surfacePoints)
        # find three closest points of surfacePoints to points
        [d, inds] = nbrs.kneighbors(points)
        # points of triangle of three closest points
        A = surfacePoints[inds[:, 0], :]
        B = surfacePoints[inds[:, 1], :]
        C = surfacePoints[inds[:, 2], :]

        # convert points to barycentric co-ordinates
        BCs = self.cartesian2barycentric(A, B, C, points)
        # reconstruct location in plane where A , B and C lie
        out = self.barycentric2cartesian(A, B, C, BCs)
        return out

    def mydot(self, A, B, axis=1):
        """Calculates the scalar dot product of corresponding rows or columns of A and B
        INPUTS:
            A - A p x q array
            B - A p x q array
            axis - (default = 1) if 1 computes dot product for vectors in rows of A and b, if 0 will callulate for columns
        OUTPUTS:
            out - a vector containg the dot products
        """
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)
        out = np.sum(A * B, axis=axis)
        return out

    def barycentric2cartesian(self, A, B, C, BCs):
        out = np.zeros(A.shape)
        points = [A, B, C]
        for i in range(3):
            out = out + np.atleast_2d(BCs[:, i]).T * points[i]
        return out

    def cartesian2barycentric(self, A, B, C, P):
        """
            Converts cartesian co-ordinates (in each row of P) into braycentric co-ordinates relative to the triangle
            defined by the three co-ordinates in the corresponding row of each A, B and C
        INPUTS:
            A - a q x 3 array of co-ordinates of the first point of the triangle
            B - a q x 3 array of co-ordinates of the second point of the triangle
            C - a q x 3 array of co-ordinates of the third point of the triangle
            P - a q x 3 array of co-ordinates, to convert into barycentric co-ordinates

        OUTPUTS:
            out - a q x 3 array of barycentric co-ordinates for each point in P
        REFERENCES:
            https://en.wikipedia.org/wiki/Barycentric_coordinate_system
            http://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates"""
        v0 = B - A
        v1 = C - A
        v2 = P - A

        d00 = self.mydot(v0, v0)
        d01 = self.mydot(v0, v1)
        d11 = self.mydot(v1, v1)
        d20 = self.mydot(v2, v0)
        d21 = self.mydot(v2, v1)
        denom = d00 * d11 - d01 * d01
        beta = (d11 * d20 - d01 * d21) / denom
        gamma = (d00 * d21 - d01 * d20) / denom
        alpha = 1.0 - gamma - beta
        out = np.concatenate((np.atleast_2d(alpha).T, np.atleast_2d(beta).T,
                              np.atleast_2d(gamma).T),
                             axis=1)
        return out



class UnitNormaliseScalar(object):
    def __init__(self, key = 'x'):
        self.key = key

    def __call__(self, data):
        data[self.key] = ((data[self.key] - data[self.key].min()) /
                         (data[self.key].max() - data[self.key].min()))

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.key)
