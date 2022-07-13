from collections import deque
import math
import os

import torch
import torch.nn as nn

from pyrutils.itertools import negative_range
from pyrutils.torch.distributions import straight_through_gumbel_sigmoid, straight_through_estimator
from pyrutils.torch.general import cat_valid_tensors
from pyrutils.torch.models import build_mlp
from pyrutils.torch.models_gcn import Geo_gcn


class BimanualBaseline(nn.Module):
    def __init__(self, input_size: tuple, num_classes: tuple, hidden_size: int = 128, bidirectional: bool = True,
                 with_message_passing: bool = True, bias: bool = True):
        super(BimanualBaseline, self).__init__()
        human_input_size, object_input_size = input_size
        num_subactivities, _ = num_classes
        self.with_message_passing = with_message_passing

        self.human_embedding_mlp = build_mlp([human_input_size, hidden_size], ['relu'], bias=bias)
        self.object_embedding_mlp = build_mlp([object_input_size, hidden_size], ['relu'], bias=bias)
        self.human_bd_rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, bias=bias, batch_first=True,
                                   bidirectional=bidirectional)
        self.object_bd_rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, bias=bias, batch_first=True,
                                    bidirectional=bidirectional)
        recognition_input_size = hidden_size
        if with_message_passing:
            recognition_input_size *= 2
        if bidirectional:
            recognition_input_size *= 2
        self.human_recognition_mlp = build_mlp([recognition_input_size, num_subactivities],
                                               [{'name': 'logsoftmax', 'dim': -1}], bias=bias)

    def forward(self, x_human, x_objects, objects_mask):
        """Forward input tensors through BimanualBaseline.

        Arguments:
            x_human - Tensor of shape (batch_size, num_steps, num_humans, human_feature_size) containing frame-level
                features for the human(s) in the video.
            x_objects - Tensor of shape (batch_size, num_steps, num_objects, object_feature_size) containing
                frame-level features for the objects in the video.
            objects_mask - Binary tensor of shape (batch_size, num_objects) indicating whether an object is real or
                virtual. Virtual objects are used to enable batched operations.
        """
        # Initial Embeddings
        x_human, x_objects = self.human_embedding_mlp(x_human), self.object_embedding_mlp(x_objects)
        # Frame-level BiRNNs
        hx_hfr = self._process_frame_level_rnn(x_human, self.human_bd_rnn)
        hx_ofr = self._process_frame_level_rnn(x_objects, self.object_bd_rnn)
        # Objects -> Humans message
        if self.with_message_passing:
            # Pool Objects
            objects_mask = torch.unsqueeze(torch.unsqueeze(objects_mask, dim=1), dim=-1)
            hx_ofr = hx_ofr * objects_mask
            hx_ofr = torch.sum(hx_ofr, dim=2, keepdim=True)
            num_real_objects = torch.clamp(torch.sum(objects_mask, dim=2, keepdim=True), min=1.0)
            hx_ofr = hx_ofr / num_real_objects
            # Merge objects with humans
            num_humans = x_human.size(2)
            hx_ofr = torch.repeat_interleave(hx_ofr, repeats=num_humans, dim=2)
            hx = torch.cat([hx_hfr, hx_ofr], dim=-1)
        else:
            hx = hx_hfr
        y_human_recognition = self.human_recognition_mlp(hx).permute(0, 3, 1, 2).contiguous()
        return [y_human_recognition]

    @staticmethod
    def _process_frame_level_rnn(x, rnn):
        """Process frame-level RNNs.

        Arguments:
            x - Tensor of shape (batch_size, num_steps, num_entities, hidden_size) containing the frame-level input
                features of all entities.
            rnn - PyTorch RNN module to process every entity in x.
        Returns:
            A tensor of shape (batch_size, num_steps, num_entities, 2 * hidden_size) containing the frame-level hidden
            states of the input entities.
        """
        h_fr, num_entities = [], x.size(2)
        for e in range(num_entities):
            h_fe, _ = rnn(x[:, :, e])
            h_fr.append(h_fe)
        h_fr = torch.stack(h_fr, dim=2)
        return h_fr


class CAD120Baseline(nn.Module):
    def __init__(self, input_size: tuple, num_classes: tuple, hidden_size: int = 128, bidirectional: bool = True,
                 with_message_passing: bool = True, bias: bool = True):
        super(CAD120Baseline, self).__init__()
        human_input_size, object_input_size = input_size
        num_subactivities, num_affordances = num_classes
        self.with_message_passing = with_message_passing

        self.human_embedding_mlp = build_mlp([human_input_size, hidden_size], ['relu'], bias=bias)
        self.object_embedding_mlp = build_mlp([object_input_size, hidden_size], ['relu'], bias=bias)
        self.human_bd_rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, bias=bias, batch_first=True,
                                   bidirectional=bidirectional)
        self.object_bd_rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, bias=bias, batch_first=True,
                                    bidirectional=bidirectional)
        recognition_input_size = hidden_size
        if with_message_passing:
            recognition_input_size *= 2
        if bidirectional:
            recognition_input_size *= 2
        self.human_recognition_mlp = build_mlp([recognition_input_size, num_subactivities],
                                               [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
        self.object_recognition_mlp = build_mlp([recognition_input_size, num_affordances],
                                                [{'name': 'logsoftmax', 'dim': -1}], bias=bias)

    def forward(self, x_human, x_objects, objects_mask):
        """Forward input tensors through CAD120Baseline.

        Arguments:
            x_human - Tensor of shape (batch_size, num_steps, num_humans, human_feature_size) containing frame-level
                features for the human(s) in the video.
            x_objects - Tensor of shape (batch_size, num_steps, num_objects, object_feature_size) containing
                frame-level features for the objects in the video.
            objects_mask - Binary tensor of shape (batch_size, num_objects) indicating whether an object is real or
                virtual. Virtual objects are used to enable batched operations.
        """
        # Initial Embeddings
        x_human, x_objects = self.human_embedding_mlp(x_human), self.object_embedding_mlp(x_objects)
        # Frame-level BiRNNs
        hx_hfr = self._process_frame_level_rnn(x_human, self.human_bd_rnn)
        hx_ofr = self._process_frame_level_rnn(x_objects, self.object_bd_rnn)
        # Objects -> Human message
        if self.with_message_passing:
            # Pool Objects
            objects_mask = torch.unsqueeze(torch.unsqueeze(objects_mask, dim=1), dim=-1)
            hx_ofm = hx_ofr * objects_mask
            hx_ofm = torch.sum(hx_ofm, dim=2, keepdim=True)
            num_real_objects = torch.clamp(torch.sum(objects_mask, dim=2, keepdim=True), min=1.0)
            hx_ofm = hx_ofm / num_real_objects
            # Merge objects with humans
            num_humans = x_human.size(2)
            hx_ofm = torch.repeat_interleave(hx_ofm, repeats=num_humans, dim=2)
            hx_h = torch.cat([hx_hfr, hx_ofm], dim=-1)
        else:
            hx_h = hx_hfr
        y_human_recognition = self.human_recognition_mlp(hx_h).permute(0, 3, 1, 2).contiguous()
        # Human -> Object message
        if self.with_message_passing:
            # Pool Humans
            hx_hfm = torch.sum(hx_hfr, dim=2, keepdim=True)
            # Merge humans with objects
            num_objects = x_objects.size(2)
            hx_hfm = torch.repeat_interleave(hx_hfm, repeats=num_objects, dim=2)
            hx_o = torch.cat([hx_ofr, hx_hfm], dim=-1)
        else:
            hx_o = hx_ofr
        y_object_recognition = self.object_recognition_mlp(hx_o).permute(0, 3, 1, 2).contiguous()
        return [y_human_recognition, y_object_recognition]

    @staticmethod
    def _process_frame_level_rnn(x, rnn):
        """Process frame-level RNNs.

        Arguments:
            x - Tensor of shape (batch_size, num_steps, num_entities, hidden_size) containing the frame-level input
                features of all entities.
            rnn - PyTorch RNN module to process every entity in x.
        Returns:
            A tensor of shape (batch_size, num_steps, num_entities, 2 * hidden_size) containing the frame-level hidden
            states of the input entities.
        """
        h_fr, num_entities = [], x.size(2)
        for e in range(num_entities):
            h_fe, _ = rnn(x[:, :, e])
            h_fr.append(h_fe)
        h_fr = torch.stack(h_fr, dim=2)
        return h_fr


class TGGCN(nn.Module):
    def __init__(self, input_size: tuple, num_classes: tuple, hidden_size: int = 128,
                 discrete_networks_num_layers: int = 1, discrete_optimization_strategy: str = 'gumbel-sigmoid',
                 filter_discrete_updates: bool = False, gcn_node: int = 26,
                 message_humans_to_human: bool = True, message_human_to_objects: bool = True,
                 message_objects_to_human: bool = True, message_objects_to_object: bool = True,
                 message_geometry_to_objects: bool = True, message_geometry_to_human: bool = False,
                 message_segment: bool = False, message_type: str = 'relational', message_granularity: str = 'specific',
                 message_aggregation: str = 'attention', attention_style: str = 'concat',
                 object_segment_update_strategy: str = 'independent', update_segment_threshold: float = 0.5,
                 add_segment_length: bool = False, add_time_position: bool = False, time_position_strategy: str = 's',
                 positional_encoding_style: str = 'embedding', cat_level_states: bool = False,
                 share_level_mlps: bool = False, bias: bool = True):
        """Frame-level Human-object RNN with Smoothing.

        Arguments:
            input_size - A 2-element tuple containing the input sizes of the human features and the object features.
            num_classes - A 2-element tuple containing the number of sub-activity classes and the number of object
                affordance classes.
            hidden_size - A positive integer representing the hidden size of the model's internal layers.
            discrete_networks_num_layers - Number of layers for the MLPs that learn discrete decisions.
            discrete_optimization_strategy - Strategy to use when updating discrete components of the model.
                Either 'straight-through' or 'gumbel-sigmoid'. Abbreviations are also permitted: 'st' for
                straight-through and 'gs' for 'gumbel-sigmoid'.
            filter_discrete_updates - Whether to apply a local max filter to the discrete updates or not. This avoids
                situations where we have successive discrete updates.
            message_humans_to_human - Whether to send a message from the humans to another human. Only meaningful for
                datasets with multiple humans.
            message_human_to_objects - Whether to send a message from the human to the objects or not.
            message_objects_to_human - Whether to send a message from the objects to the human or not.
            message_objects_to_object - Whether to send a message from the objects to another object or not.
            message_segment - Whether to append the segment-level hidden state of an entity as an extra message to
                other entities.
            message_type - Either 'relational' ('v1') or 'non-relational' ('v2'). Only meaningful if sending messages.
            message_granularity - Either 'generic' ('v1') or 'specific' ('v2'). Only meaningful if message_type is
                'non-relational'.
            message_aggregation - For messages with multiple sources for a single receiver, how to aggregate them?
                Only meaningful for message_type 'non-relational'.
            attention_style - In case message aggregation is 'attention', what sort of attention to use?
            object_segment_update_strategy - When learning to segment the sub-activities and affordances, do we make
                the segmentation of the affordances the same as the sub-activities ('same_as_human') or 'independent'
                of the sub-activity decision? Furthermore, if it is an independent decision, do we make it conditional
                on the sub-activity decision ('conditional_on_human')? Thus, one of 'same_as_human', 'independent', or
                'conditional_on_human'. Abbreviations are permitted: 'sah' for 'same_as_human', 'ind' for
                'independent', and 'coh' for 'conditional_on_human'.
            update_segment_threshold - Threshold for the segment-level hard binary decision.
            add_segment_length - Add absolute segment length to the segment-level RNNs.
            add_time_position - Add absolute time position to the segment-level RNNs.
            time_position_strategy - Add time position to the segment-level input ['s'] or to the discrete update
                input ['u'].
            positional_encoding_style - Either 'embedding' ['e'] or 'periodic' ['p'].
            cat_level_states - If True, concatenate first and second level RNNs state before passing it to the
                labeling MLPs.
            share_level_mlps - If True, share the predictions MLPs of first and second levels.
            bias - Whether to include a bias term in the internal layers of the model.
        """
        super(TGGCN, self).__init__()
        human_input_size, object_input_size = input_size
        num_subactivities, num_affordances = num_classes
        self.discrete_optimization_strategy = discrete_optimization_strategy
        self.filter_discrete_updates = filter_discrete_updates
        self.gcn_node = gcn_node
        self.message_humans_to_human = message_humans_to_human
        self.message_human_to_objects = message_human_to_objects
        self.message_objects_to_human = message_objects_to_human
        self.message_objects_to_object = message_objects_to_object
        self.message_geometry_to_objects = message_geometry_to_objects
        self.message_geometry_to_human = message_geometry_to_human
        self.message_segment = message_segment
        self.message_type = message_type
        self.message_granularity = message_granularity
        self.message_aggregation = message_aggregation
        self.attention_style = attention_style
        self.object_segment_update_strategy = object_segment_update_strategy
        self.update_segment_threshold = update_segment_threshold
        self.add_segment_length = add_segment_length
        self.add_time_position = add_time_position
        self.time_position_strategy = time_position_strategy
        self.positional_encoding_style = positional_encoding_style
        self.cat_level_states = cat_level_states
        # Shared
        if add_time_position and positional_encoding_style in {'e', 'embedding'}:
            self.time_position_mlp = build_mlp([1, hidden_size], activations=['relu'], bias=bias)
        if add_segment_length and positional_encoding_style in {'e', 'embedding'}:
            self.segment_length_mlp = build_mlp([1, hidden_size], activations=['relu'], bias=bias)

        # geometry
        self.geometry_embedding_gcn = Geo_gcn(self.gcn_node, 4, 128)
        self.geometry_embedding_mlp = build_mlp([self.gcn_node*128, 2048, hidden_size], ['relu','relu'], bias=bias)
        self.geometry_bd_rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, bias=bias, batch_first=True,
                                      bidirectional=True)
        self.geometry_bd_embedding_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)


        # Human
        self.human_embedding_mlp = build_mlp([2048, hidden_size], ['relu'], bias=bias)
        self.human_bd_rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, bias=bias, batch_first=True,
                                   bidirectional=True)
        self.human_bd_embedding_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
        human_segment_input_size = hidden_size
        if message_humans_to_human:
            human_segment_input_size += hidden_size
            if message_segment:
                human_segment_input_size += hidden_size
        if message_geometry_to_human:
            human_segment_input_size += hidden_size
        #    if message_segment:
        #        human_segment_input_size += hidden_size
        if message_objects_to_human:
            human_segment_input_size += hidden_size
            if message_segment:
                human_segment_input_size += hidden_size
        if add_time_position and time_position_strategy == 's':
            human_segment_input_size += hidden_size
        if add_segment_length:
            human_segment_input_size += hidden_size
        self.human_segment_rnn_fcell = nn.GRUCell(human_segment_input_size, hidden_size, bias=bias)
        self.human_segment_rnn_bcell = nn.GRUCell(human_segment_input_size, hidden_size, bias=bias)

        # Object
        self.object_embedding_mlp = build_mlp([object_input_size, hidden_size], ['relu'], bias=bias)
        self.object_bd_rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, bias=bias, batch_first=True,
                                    bidirectional=True)
        self.object_bd_embedding_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
        object_segment_input_size = hidden_size
        if message_geometry_to_objects:
            object_segment_input_size += hidden_size
        #    if message_segment:
        #        object_segment_input_size += hidden_size
        if message_human_to_objects:
            object_segment_input_size += hidden_size
            if message_segment:
                object_segment_input_size += hidden_size
        if message_objects_to_object:
            object_segment_input_size += hidden_size
            if message_segment:
                object_segment_input_size += hidden_size
        if add_time_position and time_position_strategy == 's':
            object_segment_input_size += hidden_size
        if add_segment_length:
            object_segment_input_size += hidden_size
        self.object_segment_rnn_fcell = nn.GRUCell(object_segment_input_size, hidden_size, bias=bias)
        self.object_segment_rnn_bcell = nn.GRUCell(object_segment_input_size, hidden_size, bias=bias)
        # Messages
        # Human(s) to Human
        if message_humans_to_human:
            if message_type in {'v1', 'relational'}:
                self.human_human_pairwise_relation_mlp = build_mlp([4 * hidden_size, hidden_size], ['relu'], bias=bias)
                self.human_human_full_relation_mlp = build_mlp([hidden_size, hidden_size], ['relu'], bias=bias)
                if message_segment:
                    self.human_human_segment_pairwise_relation_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'],
                                                                               bias=bias)
                    self.human_human_segment_full_relation_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                           bias=bias)
            else:  # v2 or non-relational
                if message_granularity in {'v1', 'generic'}:
                    self.humans_to_human_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
                    if message_segment:
                        self.humans_to_human_segment_message_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                             bias=bias)
                else:  # v2 or specific
                    self.humans_to_human_message_mlp = build_mlp([4 * hidden_size, hidden_size], ['relu'], bias=bias)
                    if message_segment:
                        self.humans_to_human_segment_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'],
                                                                             bias=bias)
                if message_aggregation in {'att', 'attention'}:
                    if attention_style in {'v4', 'general'}:
                        self.humans_to_human_message_att_mlp = nn.Bilinear(2 * hidden_size, 2 * hidden_size, 1,
                                                                           bias=bias)
                        if message_segment:
                            self.humans_to_human_segment_message_att_mlp = nn.Bilinear(hidden_size, hidden_size, 1,
                                                                                       bias=bias)
                    else:
                        self.humans_to_human_message_att_mlp = build_mlp([4 * hidden_size, 1], ['relu'], bias=bias)
                        if message_segment:
                            self.humans_to_human_segment_message_att_mlp = build_mlp([2 * hidden_size, 1], ['relu'],
                                                                                     bias=bias)
        # Human(s) to Object
        if message_human_to_objects:
            if message_type in {'v1', 'relational'}:
                self.object_human_pairwise_relation_mlp = build_mlp([4 * hidden_size, hidden_size], ['relu'], bias=bias)
                self.object_human_full_relation_mlp = build_mlp([hidden_size, hidden_size], ['relu'], bias=bias)
                if message_segment:
                    self.object_human_segment_pairwise_relation_mlp = build_mlp([2 * hidden_size, hidden_size],
                                                                                ['relu'], bias=bias)
                    self.object_human_segment_full_relation_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                            bias=bias)
            else:  # v2 or non-relational
                if message_granularity in {'v1', 'generic'}:
                    self.human_to_object_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
                    if message_segment:
                        self.human_to_object_segment_message_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                             bias=bias)
                else:  # v2 or specific
                    self.human_to_object_message_mlp = build_mlp([4 * hidden_size, hidden_size], ['relu'], bias=bias)
                    if message_segment:
                        self.human_to_object_segment_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'],
                                                                             bias=bias)
                if message_aggregation in {'att', 'attention'}:
                    if attention_style in {'v4', 'general'}:
                        self.humans_to_object_message_att_mlp = nn.Bilinear(2 * hidden_size, 2 * hidden_size, 1,
                                                                            bias=bias)
                        if message_segment:
                            self.humans_to_object_segment_message_att_mlp = nn.Bilinear(hidden_size, hidden_size, 1,
                                                                                        bias=bias)
                    else:
                        self.humans_to_object_message_att_mlp = build_mlp([4 * hidden_size, 1], ['relu'], bias=bias)
                        if message_segment:
                            self.humans_to_object_segment_message_att_mlp = build_mlp([2 * hidden_size, 1], ['relu'],
                                                                                      bias=bias)
        # Objects to Human
        if message_objects_to_human:
            if message_type in {'v1', 'relational'}:
                self.human_object_pairwise_relation_mlp = build_mlp([4 * hidden_size, hidden_size], ['relu'], bias=bias)
                self.human_object_full_relation_mlp = build_mlp([hidden_size, hidden_size], ['relu'], bias=bias)
                if message_segment:
                    self.human_object_segment_pairwise_relation_mlp = build_mlp([2 * hidden_size, hidden_size],
                                                                                ['relu'], bias=bias)
                    self.human_object_segment_full_relation_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                            bias=bias)
            else:  # v2 or non-relational
                if message_granularity in {'v1', 'generic'}:
                    self.objects_to_human_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
                    if message_segment:
                        self.objects_to_human_segment_message_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                              bias=bias)
                else:  # v2 or specific
                    self.objects_to_human_message_mlp = build_mlp([4 * hidden_size, hidden_size], ['relu'], bias=bias)
                    if message_segment:
                        self.objects_to_human_segment_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'],
                                                                              bias=bias)
                if message_aggregation in {'att', 'attention'}:
                    if attention_style in {'v4', 'general'}:
                        self.objects_to_human_message_att_mlp = nn.Bilinear(2 * hidden_size, 2 * hidden_size, 1,
                                                                            bias=bias)
                        if message_segment:
                            self.objects_to_human_segment_message_att_mlp = nn.Bilinear(hidden_size, hidden_size, 1,
                                                                                        bias=bias)
                    else:
                        self.objects_to_human_message_att_mlp = build_mlp([4 * hidden_size, 1], ['relu'], bias=bias)
                        if message_segment:
                            self.objects_to_human_segment_message_att_mlp = build_mlp([2 * hidden_size, 1], ['relu'],
                                                                                      bias=bias)
        # Objects to Object
        if message_objects_to_object:
            if message_type in {'v1', 'relational'}:
                self.object_object_pairwise_relation_mlp = build_mlp([4 * hidden_size, hidden_size], ['relu'],
                                                                     bias=bias)
                self.object_object_full_relation_mlp = build_mlp([hidden_size, hidden_size], ['relu'], bias=bias)
                if message_segment:
                    self.object_object_segment_pairwise_relation_mlp = build_mlp([2 * hidden_size, hidden_size],
                                                                                 ['relu'], bias=bias)
                    self.object_object_segment_full_relation_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                             bias=bias)
            else:  # v2 or non-relational
                if message_granularity in {'v1', 'generic'}:
                    self.objects_to_object_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
                    if message_segment:
                        self.objects_to_object_segment_message_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                               bias=bias)
                else:  # v2 or specific
                    self.objects_to_object_message_mlp = build_mlp([4 * hidden_size, hidden_size], ['relu'], bias=bias)
                    if message_segment:
                        self.objects_to_object_segment_message_mlp = build_mlp([2 * hidden_size, hidden_size],
                                                                               ['relu'], bias=bias)
                if message_aggregation in {'att', 'attention'}:
                    if attention_style in {'v4', 'general'}:
                        self.objects_to_object_message_att_mlp = nn.Bilinear(2 * hidden_size, 2 * hidden_size, 1,
                                                                             bias=bias)
                        if message_segment:
                            self.objects_to_object_segment_message_att_mlp = nn.Bilinear(hidden_size, hidden_size, 1,
                                                                                         bias=bias)
                    else:
                        self.objects_to_object_message_att_mlp = build_mlp([4 * hidden_size, 1], ['relu'], bias=bias)
                        if message_segment:
                            self.objects_to_object_segment_message_att_mlp = build_mlp([2 * hidden_size, 1], ['relu'],
                                                                                       bias=bias)
        # geometry(s) to Human
        if message_geometry_to_human:
            if message_type in {'v1', 'relational'}:
                self.human_geometry_pairwise_relation_mlp = build_mlp([4 * hidden_size, hidden_size], ['relu'], bias=bias)
                self.human_geometry_full_relation_mlp = build_mlp([hidden_size, hidden_size], ['relu'], bias=bias)
                if message_segment:
                    self.human_geometry_segment_pairwise_relation_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'],
                                                                               bias=bias)
                    self.human_geometry_segment_full_relation_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                           bias=bias)
            else:  # v2 or non-relational
                if message_granularity in {'v1', 'generic'}:
                    self.geometry_to_human_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
                    if message_segment:
                        self.geometry_to_human_segment_message_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                             bias=bias)
                else:  # v2 or specific
                    self.geometry_to_human_message_mlp = build_mlp([4 * hidden_size, hidden_size], ['relu'], bias=bias)
                    if message_segment:
                        self.geometry_to_human_segment_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'],
                                                                             bias=bias)
                if message_aggregation in {'att', 'attention'}:
                    if attention_style in {'v4', 'general'}:
                        self.geometry_to_human_message_att_mlp = nn.Bilinear(2 * hidden_size, 2 * hidden_size, 1,
                                                                           bias=bias)
                        if message_segment:
                            self.geometry_to_human_segment_message_att_mlp = nn.Bilinear(hidden_size, hidden_size, 1,
                                                                                       bias=bias)
                    else:
                        self.geometry_to_human_message_att_mlp = build_mlp([4 * hidden_size, 1], ['relu'], bias=bias)
                        if message_segment:
                            self.geometry_to_human_segment_message_att_mlp = build_mlp([2 * hidden_size, 1], ['relu'],
                                                                                     bias=bias)
        # geometry(s) to Object
        if message_geometry_to_objects:
            if message_type in {'v1', 'relational'}:
                self.object_geometry_pairwise_relation_mlp = build_mlp([4 * hidden_size, hidden_size], ['relu'], bias=bias)
                self.object_geometry_full_relation_mlp = build_mlp([hidden_size, hidden_size], ['relu'], bias=bias)
                if message_segment:
                    self.object_geometry_segment_pairwise_relation_mlp = build_mlp([2 * hidden_size, hidden_size],
                                                                                ['relu'], bias=bias)
                    self.object_geometry_segment_full_relation_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                            bias=bias)
            else:  # v2 or non-relational
                if message_granularity in {'v1', 'generic'}:
                    self.geometry_to_object_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'], bias=bias)
                    if message_segment:
                        self.geometry_to_object_segment_message_mlp = build_mlp([hidden_size, hidden_size], ['relu'],
                                                                             bias=bias)
                else:  # v2 or specific
                    self.geometry_to_object_message_mlp = build_mlp([4 * hidden_size, hidden_size], ['relu'], bias=bias)
                    if message_segment:
                        self.geometry_to_object_segment_message_mlp = build_mlp([2 * hidden_size, hidden_size], ['relu'],
                                                                             bias=bias)
                if message_aggregation in {'att', 'attention'}:
                    if attention_style in {'v4', 'general'}:
                        self.geometry_to_object_message_att_mlp = nn.Bilinear(2 * hidden_size, 2 * hidden_size, 1,
                                                                            bias=bias)
                        if message_segment:
                            self.geometry_to_object_segment_message_att_mlp = nn.Bilinear(hidden_size, hidden_size, 1,
                                                                                        bias=bias)
                    else:
                        self.geometry_to_object_message_att_mlp = build_mlp([4 * hidden_size, 1], ['relu'], bias=bias)
                        if message_segment:
                            self.geometry_to_object_segment_message_att_mlp = build_mlp([2 * hidden_size, 1], ['relu'],
                                                                                      bias=bias)

        # Discrete MLPs
        update_human_segment_input_size = 2 * hidden_size
        if message_humans_to_human:
            update_human_segment_input_size += hidden_size
        if message_objects_to_human:
            update_human_segment_input_size += hidden_size
        if message_geometry_to_human:
            update_human_segment_input_size += hidden_size
        if add_time_position and time_position_strategy == 'u':
            update_human_segment_input_size += hidden_size
        num_discrete_hidden_layers = discrete_networks_num_layers - 1
        dims = [update_human_segment_input_size] + [hidden_size] * num_discrete_hidden_layers + [1]
        activations = ['relu'] * num_discrete_hidden_layers + ['sigmoid']
        self.update_human_segment_mlp = build_mlp(dims, activations, bias=bias)

        if object_segment_update_strategy not in {'same_as_human', 'sah'}:
            update_object_segment_input_size = 2 * hidden_size
            if message_human_to_objects:
                update_object_segment_input_size += hidden_size
            if message_objects_to_object:
                update_object_segment_input_size += hidden_size
            if message_geometry_to_objects:
                update_object_segment_input_size += hidden_size
            if add_time_position and time_position_strategy == 'u':
                update_object_segment_input_size += hidden_size
            dims = [update_object_segment_input_size] + [hidden_size] * num_discrete_hidden_layers + [1]
            self.update_object_segment_mlp = build_mlp(dims, activations, bias=bias)

        # geometry segment update is the same as human

        # Recognition/Prediction MLPs
        label_mlps_input_size = 2 * hidden_size
        if cat_level_states:
            label_mlps_input_size += 2 * hidden_size
        self.human_recognition_mlp = build_mlp([label_mlps_input_size, num_subactivities],
                                               [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
        self.human_prediction_mlp = build_mlp([label_mlps_input_size, num_subactivities],
                                              [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
        if num_affordances is not None:
            self.object_recognition_mlp = build_mlp([label_mlps_input_size, num_affordances],
                                                    [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
            self.object_prediction_mlp = build_mlp([label_mlps_input_size, num_affordances],
                                                   [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
        if share_level_mlps and not cat_level_states:
            self.human_frame_recognition_mlp = self.human_recognition_mlp
            self.human_frame_prediction_mlp = self.human_prediction_mlp
            if num_affordances is not None:
                self.object_frame_recognition_mlp = self.object_recognition_mlp
                self.object_frame_prediction_mlp = self.object_prediction_mlp
        else:
            self.human_frame_recognition_mlp = build_mlp([2 * hidden_size, num_subactivities],
                                                         [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
            self.human_frame_prediction_mlp = build_mlp([2 * hidden_size, num_subactivities],
                                                        [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
            if num_affordances is not None:
                self.object_frame_recognition_mlp = build_mlp([2 * hidden_size, num_affordances],
                                                              [{'name': 'logsoftmax', 'dim': -1}], bias=bias)
                self.object_frame_prediction_mlp = build_mlp([2 * hidden_size, num_affordances],
                                                             [{'name': 'logsoftmax', 'dim': -1}], bias=bias)

        # geometry no need predict or segment

    def forward(self, x_human, x_objects, objects_mask, human_segmentation=None, objects_segmentation=None,
                human_human_distances=None, human_object_distances=None, object_object_distances=None,
                steps_per_example=None, inspect_model=False):
        """Forward input tensors through the FrameLevelHumanObjectRNN.

        Arguments:
            x_human - Tensor of shape (batch_size, num_steps, num_humans, human_feature_size) containing frame-level
                features for the human(s) in the video.
            x_objects - Tensor of shape (batch_size, num_steps, num_objects, object_feature_size) containing
                frame-level features for the objects in the video.
            objects_mask - Binary tensor of shape (batch_size, num_objects) indicating whether an object is real or
                virtual. Virtual objects are used to enable batched operations.
            human_segmentation - Binary tensor of shape (batch_size, num_steps, num_humans) indicating whether a frame
                is the last frame of the sub-activity segment (1) or not (0). If None, the model learns the
                segmentation itself.
            objects_segmentation - Binary tensor of shape (batch_size, num_steps, num_objects) indicating, for each
                object, whether a frame is the last frame of the affordance segment (1) or not (0). If None, the model
                learns the segmentation itself.
            human_human_distances - An optional tensor of shape (batch_size, num_steps, num_humans, num_humans)
                containing the distances between humans. Only meaningful if doing attention, in which case it replaces
                the computation of the attention weights.
            human_object_distances - An optional tensor of shape (batch_size, num_steps, num_humans, num_objects)
                containing the distances between humans and objects. Only meaningful if doing attention, in which case
                it replaces the computation of the attention weights.
            object_object_distances - An optional tensor of shape (batch_size, num_steps, num_objects, num_objects)
                containing the distances between objects. Only meaningful if doing attention, in which case it replaces
                the computation of the attention weights.
            steps_per_example - A tensor of shape (batch_size,) containing the real number of steps of each video.
            inspect_model - If True, return some other things such as attention scores in addition to the usual outputs.
        Returns:
            Eight tensors: sub_activity_hard_segmentation, affordance_hard_segmentation,
            sub_activity_soft_segmentation, affordance_soft_segmentation, sub_activity_recognition,
            sub_activity_prediction, affordance_recognition, and affordance_prediction. sub_activity_hard_segmentation
            is a tensor of shape (batch_size, num_steps), affordance_hard_segmentation is a tensor of shape
            (batch_size, num_steps, num_objects), sub_activity_recognition and sub_activity_prediction are tensors of
            shape (batch_size, num_sub_activity_classes, num_steps), and affordance_recognition and
            affordance_prediction are tensors of shape (batch_size, num_affordance_classes, num_steps, num_objects).
            sub_activity_soft_segmentation and affordance_soft_segmentation have the same shape as the hard
            counterparts.
        """
        num_humans, num_objects = x_human.size(2), x_objects.size(2)
        xx_hs, xx_os = [[] for _ in range(num_humans)], [[] for _ in range(num_objects)]
        ux_hs, ux_os = [[] for _ in range(num_humans)], [[] for _ in range(num_objects)]  # Hard
        ux_hss, ux_oss = [[] for _ in range(num_humans)], [[] for _ in range(num_objects)]  # Soft
        ax_hf = [[] for _ in range(num_humans)]

        ### gcn ###
        if x_human.shape[3] == 2124:                   # CAD120
            x_human, x_geometry = torch.split(x_human, [2048, 76], dim=-1)
            x_geometry = x_geometry.squeeze(2)
        elif x_human.shape[3] == 2168:                 # Bimacs
            x_human, x_geometry = torch.split(x_human, [2048, 120], dim=-1)
            x_geometry = x_geometry[:, :, 0, :]
        else:                                          # MPHOI (2152)
            x_human, x_geometry = torch.split(x_human, [2048, 104], dim=-1)
            x_geometry = x_geometry[:, :, 0, :]
        bs, t, vw = x_geometry.size()
        x_geometry = x_geometry.view(bs, t, vw // 4, 4)
        x_geometry = x_geometry.permute(0, 3, 2, 1).contiguous()
        x_geometry = self.geometry_embedding_gcn(x_geometry)
        x_geometry = x_geometry.unsqueeze(2)
        x_geometry = x_geometry.view(bs, t, 1, x_geometry.shape[1] * x_geometry.shape[3])
        x_human, x_objects, x_geometry = self.human_embedding_mlp(x_human), self.object_embedding_mlp(x_objects), self.geometry_embedding_mlp(x_geometry)

        # Frame-level BiRNNs
        h_hf, h_hfr = self._process_frame_level_rnn(x_human, self.human_bd_rnn, self.human_bd_embedding_mlp)
        h_of, h_ofr = self._process_frame_level_rnn(x_objects, self.object_bd_rnn, self.object_bd_embedding_mlp)
        h_sf, h_sfr = self._process_frame_level_rnn(x_geometry, self.geometry_bd_rnn, self.geometry_bd_embedding_mlp)
        # Optional: add time position to discrete updates
        num_steps = x_human.size(1)
        ignore_division_by_number_of_steps = self.positional_encoding_style in {'p', 'periodic'}
        x_time = None
        if self.add_time_position and self.time_position_strategy == 'u':
            x_time = self._assemble_time_tensor(steps_per_example, num_steps, ignore_division_by_number_of_steps)
            if self.positional_encoding_style in {'e', 'embedding'}:
                x_time = self.time_position_mlp(x_time)
            else:
                x_time = make_periodic_embedding(x_time, self.human_segment_rnn_fcell.hidden_size)
            x_time = torch.transpose(x_time, 0, 1)
        # Frame-level processing
        for t in range(num_steps):
            try:
                x_tt = x_time[:, t]
            except TypeError:
                x_tt = None
            # Human(s)
            for h in range(num_humans):
                x_hfth = x_human[:, t, h]
                h_hfth = h_hf[:, t, h]
                m_hhth = None
                if self.message_humans_to_human:
                    try:
                        hh_dists = human_human_distances[:, t]
                    except TypeError:
                        hh_dists = None
                    m_hhth = self._humans_to_human_message(x_human[:, t], h_hf[:, t], h, hh_dists=hh_dists)
                m_ohth = None
                if self.message_objects_to_human:
                    try:
                        ho_dists = human_object_distances[:, t, h]
                    except TypeError:
                        ho_dists = None
                    m_ohth, o2h_faw = self._objects_to_human_message(x_hfth, x_objects[:, t], h_hfth, h_of[:, t],
                                                                     objects_mask, ho_dists=ho_dists)
                    ax_hf[h].append(o2h_faw)
                m_shth = None
                if self.message_geometry_to_human:
                    #try:
                    #    hs_dists = human_geometry_distances[:, t, h]
                    #except TypeError:
                    #    hs_dists = None
                    m_shth = self._geometry_to_human_message(x_geometry[:, t], x_hfth, h_sf[:, t], h_hfth,
                                                                       hs_dists=None)#hs_dists)
                try:
                    u_hsth = u_hsths = human_segmentation[:, t:t + 1, h]
                except TypeError:
                    u_hsth, u_hsths = self._update_human_segment(x_hfth, h_hfth, m_hhth, m_ohth, m_shth, x_tt)
                    if t == (num_steps - 1):
                        u_hsth[:] = 1.0
                ux_hs[h].append(u_hsth)
                ux_hss[h].append(u_hsths)
                x_hsth = cat_valid_tensors([h_hfth, m_hhth, m_ohth, m_shth], dim=-1)
                xx_hs[h].append(x_hsth)

            # Objects
            for k in range(num_objects):
                x_oftk = x_objects[:, t, k]
                h_oftk = h_of[:, t, k]
                m_hotk = None
                if self.message_human_to_objects:
                    try:
                        oh_dists = human_object_distances[:, t, :, k]
                    except TypeError:
                        oh_dists = None
                    m_hotk = self._humans_to_object_message(x_human[:, t], x_oftk, h_hf[:, t], h_oftk,
                                                            oh_dists=oh_dists)
                    m_hotk = m_hotk * objects_mask[:, k:k + 1]
                m_sotk = None
                if self.message_geometry_to_objects:
                    #try:
                    #    os_dists = geometry_object_distances[:, t, :, k]
                    #except TypeError:
                    #    os_dists = None
                    m_sotk = self._geometry_to_object_message(x_geometry[:, t], x_oftk, h_sf[:, t], h_oftk,
                                                            os_dists=None)#os_dists)
                    m_sotk = m_sotk * objects_mask[:, k:k + 1]
                m_ootk = None
                if self.message_objects_to_object:
                    try:
                        oo_dists = object_object_distances[:, t]
                    except TypeError:
                        oo_dists = None
                    m_ootk = self._objects_to_object_message(x_objects[:, t], h_of[:, t], k, objects_mask,
                                                             oo_dists=oo_dists)
                try:
                    u_ostk = u_ostks = objects_segmentation[:, t:t + 1, k]
                except TypeError:
                    u_hst = ux_hs[0][-1] if len(ux_hs) == 1 else None
                    u_hsts = ux_hss[0][-1] if len(ux_hss) == 1 else None
                    u_ostk, u_ostks = self._update_object_segment(x_oftk, h_oftk, m_hotk, m_ootk, m_sotk, u_hst, u_hsts, x_tt)
                    if t == (num_steps - 1):
                        u_ostk[:] = 1.0
                ux_os[k].append(u_ostk)
                ux_oss[k].append(u_ostks)
                x_ostk = cat_valid_tensors([h_oftk, m_hotk, m_sotk, m_ootk], dim=-1)
                xx_os[k].append(x_ostk)
        # Optional: filter discrete updates for their local maxima.
        if self.filter_discrete_updates:
            ux_hs = [filter_soft_decisions(ux_hssh, self.update_segment_threshold) for ux_hssh in ux_hss]
            ux_os = [filter_soft_decisions(ux_ossk, self.update_segment_threshold) for ux_ossk in ux_oss]
        # Optional: add position-based features
        if self.add_time_position and self.time_position_strategy == 's':
            x_time = self._assemble_time_tensor(steps_per_example, num_steps, ignore_division_by_number_of_steps)
            if self.positional_encoding_style in {'e', 'embedding'}:
                x_time = self.time_position_mlp(x_time)
            else:
                x_time = make_periodic_embedding(x_time, self.human_segment_rnn_fcell.hidden_size)
            xx_hs = [[torch.cat([x_hst, x_t], dim=-1) for x_hst, x_t in zip(x_hs, x_time)] for x_hs in xx_hs]
            xx_os = [[torch.cat([x_ost, x_t], dim=-1) for x_ost, x_t in zip(x_os, x_time)] for x_os in xx_os]
        if self.add_segment_length:
            x_hsl = self._assemble_segment_length_tensor(ux_hs, steps_per_example, ignore_division_by_number_of_steps)
            if self.positional_encoding_style in {'e', 'embedding'}:
                x_hsl = self.segment_length_mlp(x_hsl)
            else:
                x_hsl = make_periodic_embedding(x_hsl, self.human_segment_rnn_fcell.hidden_size)
            x_hsl = x_hsl.permute(2, 1, 0, 3).contiguous()
            xx_hs = [[torch.cat([x_hst, x_hslht], dim=-1) for x_hst, x_hslht in zip(x_hs, x_hslh)]
                     for x_hs, x_hslh in zip(xx_hs, x_hsl)]
            x_osl = self._assemble_segment_length_tensor(ux_os, steps_per_example, ignore_division_by_number_of_steps)
            if self.positional_encoding_style in {'e', 'embedding'}:
                x_osl = self.segment_length_mlp(x_osl)
            else:
                x_osl = make_periodic_embedding(x_osl, self.human_segment_rnn_fcell.hidden_size)
            x_osl = x_osl.permute(2, 1, 0, 3).contiguous()
            xx_os = [[torch.cat([x_ost, x_oslkt], dim=-1) for x_ost, x_oslkt in zip(x_os, x_oslk)]
                     for x_os, x_oslk in zip(xx_os, x_osl)]
        # Segment-level processing
        batch_size, hidden_size, dtype, device = x_human.size(0), x_human.size(-1), x_human.dtype, x_human.device
        hx_hsf, hx_hsb = [[] for _ in range(num_humans)], [deque() for _ in range(num_humans)]
        hx_osf, hx_osb = [[] for _ in range(num_objects)], [deque() for _ in range(num_objects)]
        ax_hsf, ax_hsb = [[] for _ in range(num_humans)], [deque() for _ in range(num_humans)]
        for tf, tb in zip(range(num_steps), negative_range(num_steps)):
            # Human(s)
            hx_hsf_cache, hx_hsb_cache = [], []
            for h in range(num_humans):
                x_hsthf = xx_hs[h][tf]
                if self.message_segment:
                    if self.message_humans_to_human:
                        try:
                            hh_dists = human_human_distances[:, tf]
                        except TypeError:
                            hh_dists = None
                        mg_hhthf = self._humans_to_human_segment_message(hx_hsf, h, batch_size, dtype, device,
                                                                         direction='forward', hh_dists=hh_dists)
                        x_hsthf = torch.cat([x_hsthf, mg_hhthf], dim=-1)
                    if self.message_objects_to_human:
                        try:
                            ho_dists = human_object_distances[:, tf, h]
                        except TypeError:
                            ho_dists = None
                        mg_ohthf, o2h_sfaw = self._objects_to_human_segment_message(hx_hsf[h], hx_osf, objects_mask,
                                                                                    'forward', ho_dists=ho_dists)
                        ax_hsf[h].append(o2h_sfaw)
                        x_hsthf = torch.cat([x_hsthf, mg_ohthf], dim=-1)
                h_hsthf = self._bidirectional_step(x_hsthf, ux_hs[h][tf], hx_hsf[h], 'human', 'forward')
                hx_hsf_cache.append(h_hsthf)
                x_hsthb = xx_hs[h][tb]
                if self.message_segment:
                    if self.message_humans_to_human:
                        try:
                            hh_dists = human_human_distances[:, tb]
                        except TypeError:
                            hh_dists = None
                        mg_hhthb = self._humans_to_human_segment_message(hx_hsb, h, batch_size, dtype, device,
                                                                         direction='backward', hh_dists=hh_dists)
                        x_hsthb = torch.cat([x_hsthb, mg_hhthb], dim=-1)
                    if self.message_objects_to_human:
                        try:
                            ho_dists = human_object_distances[:, tb, h]
                        except TypeError:
                            ho_dists = None
                        mg_ohthb, o2h_sbaw = self._objects_to_human_segment_message(hx_hsb[h], hx_osb, objects_mask,
                                                                                    'backward', ho_dists=ho_dists)
                        ax_hsb[h].appendleft(o2h_sbaw)
                        x_hsthb = torch.cat([x_hsthb, mg_ohthb], dim=-1)
                h_hsthb = self._bidirectional_step(x_hsthb, ux_hs[h][tb], hx_hsb[h], 'human', 'backward')
                hx_hsb_cache.append(h_hsthb)
            # Objects
            hx_osf_cache, hx_osb_cache = [], []
            for k in range(num_objects):
                x_ostkf = xx_os[k][tf]
                if self.message_segment:
                    if self.message_human_to_objects:
                        try:
                            oh_dists = human_object_distances[:, tf, :, k]
                        except TypeError:
                            oh_dists = None
                        mg_hotf = self._humans_to_object_segment_message(hx_osf[k], hx_hsf, batch_size, dtype, device,
                                                                         direction='forward', oh_dists=oh_dists)
                        x_ostkf = torch.cat([x_ostkf, mg_hotf], dim=-1)
                    if self.message_objects_to_object:
                        try:
                            oo_dists = object_object_distances[:, tf]
                        except TypeError:
                            oo_dists = None
                        mg_ootf = self._objects_to_object_segment_message(hx_osf, objects_mask, k, 'forward',
                                                                          oo_dists=oo_dists)
                        x_ostkf = torch.cat([x_ostkf, mg_ootf], dim=-1)
                h_ostkf = self._bidirectional_step(x_ostkf, ux_os[k][tf], hx_osf[k], 'object', 'forward')
                hx_osf_cache.append(h_ostkf)
                x_ostkb = xx_os[k][tb]
                if self.message_segment:
                    if self.message_human_to_objects:
                        try:
                            oh_dists = human_object_distances[:, tb, :, k]
                        except TypeError:
                            oh_dists = None
                        mg_hotb = self._humans_to_object_segment_message(hx_osb[k], hx_hsb, batch_size, dtype, device,
                                                                         direction='backward', oh_dists=oh_dists)
                        x_ostkb = torch.cat([x_ostkb, mg_hotb], dim=-1)
                    if self.message_objects_to_object:
                        try:
                            oo_dists = object_object_distances[:, tb]
                        except TypeError:
                            oo_dists = None
                        mg_ootb = self._objects_to_object_segment_message(hx_osb, objects_mask, k, 'backward',
                                                                          oo_dists=oo_dists)
                        x_ostkb = torch.cat([x_ostkb, mg_ootb], dim=-1)
                h_ostkb = self._bidirectional_step(x_ostkb, ux_os[k][tb], hx_osb[k], 'object', 'backward')
                hx_osb_cache.append(h_ostkb)
            # Commit updates
            for h, (h_hsf, h_hsb) in enumerate(zip(hx_hsf_cache, hx_hsb_cache)):
                hx_hsf[h].append(h_hsf)
                hx_hsb[h].appendleft(h_hsb)
            for k, (h_osf, h_osb) in enumerate(zip(hx_osf_cache, hx_osb_cache)):
                hx_osf[k].append(h_osf)
                hx_osb[k].appendleft(h_osb)
        hx_hs = [[torch.cat([h_hsthf, h_hsthb], dim=-1) for h_hsthf, h_hsthb in zip(hx_hshf, hx_hshb)]
                 for hx_hshf, hx_hshb in zip(hx_hsf, hx_hsb)]
        hx_os = [[torch.cat([h_ostkf, h_ostkb], dim=-1) for h_ostkf, h_ostkb in zip(hx_oskf, hx_oskb)]
                 for hx_oskf, hx_oskb in zip(hx_osf, hx_osb)]
        # Fix hidden states of human(s) and objects
        partial_hx_hs = []
        for h in range(num_humans):
            hx_hsh = torch.stack(hx_hs[h], dim=1)
            ux_hsh = torch.cat(ux_hs[h], dim=-1)
            hx_hsh = reorder_hidden_states(hx_hsh, ux_hsh.detach())
            partial_hx_hs.append(hx_hsh)
        hx_hs = torch.stack(partial_hx_hs, dim=2)
        partial_hx_os = []
        for k in range(num_objects):
            hx_osk = torch.stack(hx_os[k], dim=1)
            ux_osk = torch.cat(ux_os[k], dim=-1)
            hx_osk = reorder_hidden_states(hx_osk, ux_osk.detach())
            partial_hx_os.append(hx_osk)
        hx_os = torch.stack(partial_hx_os, dim=2)
        # Optional: concatenate level states
        if self.cat_level_states:
            hx_hs = torch.cat([hx_hs, h_hfr], dim=-1)
            hx_os = torch.cat([hx_os, h_ofr], dim=-1)
        # Predictions
        y_hs = torch.stack([torch.cat(ux_hsh, dim=-1) for ux_hsh in ux_hs], dim=-1)
        y_os = torch.stack([torch.cat(ux_osk, dim=-1) for ux_osk in ux_os], dim=-1)
        y_hss = torch.stack([torch.cat(ux_hssh, dim=-1) for ux_hssh in ux_hss], dim=-1)
        y_oss = torch.stack([torch.cat(ux_ossk, dim=-1) for ux_ossk in ux_oss], dim=-1)
        y_human_frame_recognition = self.human_frame_recognition_mlp(h_hfr).permute(0, 3, 1, 2).contiguous()
        y_human_frame_prediction = self.human_frame_prediction_mlp(h_hfr).permute(0, 3, 1, 2).contiguous()
        y_human_recognition = self.human_recognition_mlp(hx_hs).permute(0, 3, 1, 2).contiguous()
        y_human_prediction = self.human_prediction_mlp(hx_hs).permute(0, 3, 1, 2).contiguous()
        try:
            y_object_frame_recognition = self.object_frame_recognition_mlp(h_ofr).permute(0, 3, 1, 2).contiguous()
            y_object_frame_prediction = self.object_frame_prediction_mlp(h_ofr).permute(0, 3, 1, 2).contiguous()
            y_object_recognition = self.object_recognition_mlp(hx_os).permute(0, 3, 1, 2).contiguous()
            y_object_prediction = self.object_prediction_mlp(hx_os).permute(0, 3, 1, 2).contiguous()
        except AttributeError:
            output = [y_hs, y_hss,
                      y_human_frame_recognition, y_human_frame_prediction,
                      y_human_recognition, y_human_prediction]
        else:
            output = [y_hs, y_os, y_hss, y_oss,
                      y_human_frame_recognition, y_human_frame_prediction,
                      y_object_frame_recognition, y_object_frame_prediction,
                      y_human_recognition, y_human_prediction, y_object_recognition, y_object_prediction]
        if inspect_model:
            ax_hf = torch.stack([torch.stack(ax_hfh, dim=1) for ax_hfh in ax_hf], dim=1)
            ax_hsf = torch.stack([torch.stack(ax_hsfh, dim=1) for ax_hsfh in ax_hsf], dim=1)
            ax_hsb = torch.stack([torch.stack(list(ax_hsbh), dim=1) for ax_hsbh in ax_hsb], dim=1)
            attention_scores = [ax_hf, ax_hsf, ax_hsb]
            return output, attention_scores
        return output

    @staticmethod
    def _assemble_time_tensor(steps_per_example, max_num_steps, ignore_division_by_num_steps: bool = False):
        """Assemble a time tensor.

        Arguments:
            steps_per_example - A tensor of shape (batch_size,) containing the number of steps per example.
            max_num_steps - The number of steps in the longest video in the current data.
            ignore_division_by_num_steps - If True, do not divide by number of steps.
        Returns:
            A tensor of shape (max_num_steps, batch_size, 1) containing time features for position-based encoding.
        """
        x_time = torch.arange(1, max_num_steps + 1, dtype=steps_per_example.dtype, device=steps_per_example.device)
        x_time = torch.unsqueeze(x_time, dim=-1)
        x_time = torch.repeat_interleave(x_time, repeats=steps_per_example.size(0), dim=1)
        if not ignore_division_by_num_steps:
            x_time = x_time / steps_per_example
        x_time = torch.unsqueeze(x_time, dim=-1)
        return x_time

    def _assemble_segment_length_tensor(self, ux_s, steps_per_example,
                                        ignore_division_by_number_of_steps: bool = False):
        """Assemble segment length input for the segment-level RNNs.

        The lengths are normalised by the longest video length.

        Arguments:
            ux_s - A list containing the hard segmentation for each entity. Each list element is another list
                containing for each element segmentation per step. The elements of this inner list are tensors
                of shape (batch_size, 1).
            steps_per_example - A tensor of shape (batch_size,) containing the number of steps per example.
            ignore_division_by_number_of_steps - If True, ignore division by number of steps.
        Returns:
            A tensor of shape (batch_size, num_steps, num_entities, 1) containing the lengths of the segments.
        """
        num_steps, batch_size, dtype, device = len(ux_s[0]), ux_s[0][0].size(0), ux_s[0][0].dtype, ux_s[0][0].device
        x_time = self._assemble_time_tensor(steps_per_example, num_steps, ignore_division_by_number_of_steps)
        num_entities = len(ux_s)
        x_sl = [[] for _ in range(num_entities)]
        for e, ux_se in enumerate(ux_s):
            acc_length = torch.zeros([batch_size, 1], dtype=dtype, device=device)
            for t, (ux_set, x_t) in enumerate(zip(ux_se, x_time)):
                rel_length = ux_set * x_t
                rel_length = torch.where(rel_length.bool(), rel_length - acc_length, rel_length)
                acc_length = acc_length + rel_length
                x_sl[e].append(rel_length)
        x_sl = torch.unsqueeze(torch.stack([torch.cat(x_sle, dim=-1) for x_sle in x_sl], dim=-1), dim=-1)
        return x_sl

    @staticmethod
    def _process_frame_level_rnn(x, rnn, embedding_mlp):
        """Process frame-level RNNs.

        Arguments:
            x - Tensor of shape (batch_size, num_steps, num_entities, hidden_size) containing the frame-level input
                features of all entities.
            rnn - PyTorch RNN module to process every entity in x.
            embedding_mlp - PyTorch module to process the output of the rnn module.
        Returns:
            A tensor of shape (batch_size, num_steps, num_entities, hidden_size) containing the frame-level hidden
            states of the input entities.
        """
        h_f, num_entities = [], x.size(2)
        for e in range(num_entities):
            h_fe, _ = rnn(x[:, :, e])
            h_f.append(h_fe)
        h_fr = torch.stack(h_f, dim=2)
        h_f = embedding_mlp(h_fr)
        return h_f, h_fr

    def _humans_to_human_message(self, x_hft, h_hft, h, hh_dists=None):
        """Compute humans to human message.

        Arguments:
            x_hft - Tensor of shape (batch_size, num_humans, hidden_size) containing the frame-level features of all
                humans at time step t.
            h_hft - Tensor of shape (batch_size, num_humans, hidden_size) containing the frame-level hidden state of
                all humans at time step t.
            h - The index of the human receiving the message.
            hh_dists - An optional tensor of shape (batch_size, num_humans, num_humans) containing the distances between
                humans. Only meaningful in case message aggregation is attention, in which case the attention
                weights are dependent on the distance between humans.
        Returns:
            A tensor of shape (batch_size, hidden_size) containing a message from the other humans to human h.
        """
        x_hfth = x_hft[:, h]
        h_hfth = h_hft[:, h]
        receiver = torch.cat([x_hfth, h_hfth], dim=-1)
        x_hft = torch.cat([x_hft[:, :h], x_hft[:, h + 1:]], dim=1)
        h_hft = torch.cat([h_hft[:, :h], h_hft[:, h + 1:]], dim=1)
        senders = torch.cat([x_hft, h_hft], dim=-1)
        senders_mask = torch.ones(senders.size()[:2], dtype=senders.dtype, device=senders.device)
        if self.message_type in {'v1', 'relational'}:
            m_hhth = compute_relational_message(receiver, senders, senders_mask,
                                                f=self.human_human_full_relation_mlp,
                                                g=self.human_human_pairwise_relation_mlp)
        else:
            m_hhth = compute_non_relational_message(receiver, senders, senders_mask,
                                                    granularity=self.message_granularity,
                                                    message_fn=self.humans_to_human_message_mlp)
            if self.message_aggregation in {'mp', 'mean_pooling'}:
                num_real_senders = torch.sum(senders_mask, dim=1, keepdim=True)
                num_real_senders = torch.clamp(num_real_senders, min=1.0)
                m_hhth = torch.sum(m_hhth, dim=1) / num_real_senders
            else:
                if hh_dists is None:
                    att_weights = compute_attention_weights(receiver, senders, senders_mask,
                                                            attention_style=self.attention_style,
                                                            attention_fn=self.humans_to_human_message_att_mlp)
                else:
                    hh_dists = hh_dists[:, h]
                    hh_dists = torch.cat([hh_dists[:, :h], hh_dists[:, h + 1:]], dim=-1)
                    att_weights = compute_distance_based_attention_weights(hh_dists, senders_mask)
                att_weights = torch.unsqueeze(att_weights, dim=-1)
                m_hhth = torch.sum(att_weights * m_hhth, dim=1)
        return m_hhth

    def _humans_to_human_segment_message(self, hx_hs, h, batch_size, dtype, device, direction: str, hh_dists=None):
        """Compute segment-level message from humans to human.

        Arguments:
            hx_hs - A list containing the segment-level hidden states of all humans.
            h - The index of the human receiving the message.
            batch_size - Training batch size.
            dtype - Data type of tensors.
            device - Which device is being used for processing.
            direction - Whether the message is for the 'forward' or 'backward' RNN.
            hh_dists - An optional tensor of shape (batch_size, num_humans, num_humans) containing the distances
                between humans. Only meaningful in case message aggregation is attention, in which case the attention
                weights are dependent on the distance between humans.
        Returns:
            A tensor of shape (batch_size, hidden_size) containing a segment-level message from other humans to human h.
        """
        hidden_size = self.human_segment_rnn_fcell.hidden_size
        grab_state_fn = grab_last_state_or_zeros if direction == 'forward' else grab_first_state_or_zeros
        receiver = grab_state_fn(hx_hs[h], batch_size, hidden_size, dtype, device, detach=False)
        senders = [grab_state_fn(hx_hsm, batch_size, hidden_size, dtype, device, detach=False)
                   for hx_hsm in hx_hs[:h] + hx_hs[h + 1:]]
        senders = torch.stack(senders, dim=1)
        senders_mask = torch.ones(senders.size()[:2], dtype=senders.dtype, device=senders.device)
        if self.message_type in {'v1', 'relational'}:
            mg_hhth = compute_relational_message(receiver, senders, senders_mask,
                                                 f=self.human_human_segment_full_relation_mlp,
                                                 g=self.human_human_segment_pairwise_relation_mlp)
        else:
            mg_hhth = compute_non_relational_message(receiver, senders, senders_mask,
                                                     granularity=self.message_granularity,
                                                     message_fn=self.humans_to_human_segment_message_mlp)
            if self.message_aggregation in {'mp', 'mean_pooling'}:
                num_real_senders = torch.sum(senders_mask, dim=1, keepdim=True)
                num_real_senders = torch.clamp(num_real_senders, min=1.0)
                mg_hhth = torch.sum(mg_hhth, dim=1) / num_real_senders
            else:
                if hh_dists is None:
                    att_weights = compute_attention_weights(receiver, senders, senders_mask,
                                                            attention_style=self.attention_style,
                                                            attention_fn=self.humans_to_human_segment_message_att_mlp)
                else:
                    hh_dists = hh_dists[:, h]
                    hh_dists = torch.cat([hh_dists[:, :h], hh_dists[:, h + 1:]], dim=-1)
                    att_weights = compute_distance_based_attention_weights(hh_dists, senders_mask)
                att_weights = torch.unsqueeze(att_weights, dim=-1)
                mg_hhth = torch.sum(att_weights * mg_hhth, dim=1)
        return mg_hhth

    def _humans_to_object_message(self, x_hft, x_oftk, h_hft, h_oftk, oh_dists=None):
        """Compute human to an object message.

        For now, we assume that all humans are real.

        Arguments:
            x_hft - Tensor of shape (batch_size, num_humans, hidden_size) containing the frame-level input for the
                humans at time step t.
            x_oftk - Tensor of shape (batch_size, hidden_size) containing the frame-level input for the k-th object
                at time step t.
            h_hft - Tensor of shape (batch_size, num_humans, hidden_size) containing the frame-level hidden state for
                the humans at time step t.
            h_oftk - Tensor of shape (batch_size, hidden_size) containing frame-level hidden state of the k-th object
                at time step t.
            oh_dists - An optional tensor of shape (batch_size, num_humans) containing the distances between the k-th
                object and the humans. Only meaningful in case message aggregation is attention, in which case the
                attention weights are dependent on the distance between object and humans.
        Returns:
            A tensor of shape (batch_size, hidden_size) containing the message from the humans to the k-th object.
        """
        receiver = torch.cat([x_oftk, h_oftk], dim=-1)
        senders = torch.cat([x_hft, h_hft], dim=-1)
        senders_mask = torch.ones(senders.size()[:2], dtype=senders.dtype, device=senders.device)
        if self.message_type in {'v1', 'relational'}:
            hok_message = compute_relational_message(receiver, senders, senders_mask,
                                                     f=self.object_human_full_relation_mlp,
                                                     g=self.object_human_pairwise_relation_mlp)
        else:  # v2 or non-relational
            hok_message = compute_non_relational_message(receiver, senders, senders_mask,
                                                         granularity=self.message_granularity,
                                                         message_fn=self.human_to_object_message_mlp)
            if self.message_aggregation in {'mp', 'mean_pooling'}:
                num_real_senders = torch.sum(senders_mask, dim=1, keepdim=True)
                num_real_senders = torch.clamp(num_real_senders, min=1.0)
                hok_message = torch.sum(hok_message, dim=1) / num_real_senders
            else:
                if oh_dists is None:
                    att_weights = compute_attention_weights(receiver, senders, senders_mask,
                                                            attention_style=self.attention_style,
                                                            attention_fn=self.humans_to_object_message_att_mlp)
                else:
                    att_weights = compute_distance_based_attention_weights(oh_dists, senders_mask)
                att_weights = torch.unsqueeze(att_weights, dim=-1)
                hok_message = torch.sum(att_weights * hok_message, dim=1)
        return hok_message

    def _humans_to_object_segment_message(self, hx_hok, hx_hs, batch_size, dtype, device, direction: str,
                                          oh_dists=None):
        """Compute a segment-level message from humans to object.

        Arguments:
            hx_hok - A list containing the hidden states of object k up to time step t - 1.
            hx_hs - A list containing the hidden states of all humans up to time step t - 1.
            batch_size - Model training batch size.
            dtype - Data type of tensors.
            device - Device in which tensors are placed.
            direction - Whether the message is for the forward or backward RNN.
            oh_dists - An optional tensor of shape (batch_size, num_humans) containing the distances between the k-th
                object and the humans. Only meaningful in case message aggregation is attention, in which case the
                attention weights are dependent on the distance between object and humans.
        Returns:
            A tensor of shape (batch_size, hidden_size) containing the segment-level message from humans to object.
        """
        hidden_size = self.human_segment_rnn_fcell.hidden_size
        grab_state_fn = grab_last_state_or_zeros if direction == 'forward' else grab_first_state_or_zeros
        receiver = grab_state_fn(hx_hok, batch_size, hidden_size, dtype, device, detach=False)
        senders = [grab_state_fn(hx_hsh, batch_size, hidden_size, dtype, device, detach=False) for hx_hsh in hx_hs]
        senders = torch.stack(senders, dim=1)
        senders_mask = torch.ones(senders.size()[:2], dtype=senders.dtype, device=senders.device)
        if self.message_type in {'v1', 'relational'}:
            mg_hot = compute_relational_message(receiver, senders, senders_mask,
                                                f=self.object_human_segment_full_relation_mlp,
                                                g=self.object_human_segment_pairwise_relation_mlp)
        else:  # v2 or non-relational
            mg_hot = compute_non_relational_message(receiver, senders, senders_mask,
                                                    granularity=self.message_granularity,
                                                    message_fn=self.human_to_object_segment_message_mlp)
            if self.message_aggregation in {'mp', 'mean_pooling'}:
                num_real_senders = torch.sum(senders_mask, dim=1, keepdim=True)
                num_real_senders = torch.clamp(num_real_senders, min=1.0)
                mg_hot = torch.sum(mg_hot, dim=1) / num_real_senders
            else:
                if oh_dists is None:
                    att_weights = compute_attention_weights(receiver, senders, senders_mask,
                                                            attention_style=self.attention_style,
                                                            attention_fn=self.humans_to_object_segment_message_att_mlp)
                else:
                    att_weights = compute_distance_based_attention_weights(oh_dists, senders_mask)
                att_weights = torch.unsqueeze(att_weights, dim=-1)
                mg_hot = torch.sum(att_weights * mg_hot, dim=1)
        return mg_hot

    def _objects_to_human_message(self, x_hfth, x_oft, h_hfth, h_oft, objects_mask, ho_dists=None):
        """Compute objects to human message.

        Arguments:
            x_hfth - Tensor of shape (batch_size, hidden_size) containing the frame-level input for the human h at
                time step t.
            x_oft - Tensor of shape (batch_size, num_objects, hidden_size) containing the frame-level input for all
                objects at time step t.
            h_hfth - Tensor of shape (batch_size, hidden_size) containing the frame-level hidden state for human h at
                time step t.
            h_oft - Tensor of shape (batch_size, num_objects, hidden_size) containing the frame-level hidden state for
                all objects at time step t.
            objects_mask - Binary tensor of shape (batch_size, num_objects) specifying whether an object is real
                or virtual.
            ho_dists - An optional tensor of shape (batch_size, num_objects) containing the distance between human h
                and all objects. Only meaningful in case message aggregation is attention, in which case the attention
                weights are dependent on the distance between human and objects.
        Returns:
            A tensor of shape (batch_size, hidden_size) containing the message from objects to the human.
        """
        receiver = torch.cat([x_hfth, h_hfth], dim=-1)
        senders = torch.cat([x_oft, h_oft], dim=-1)
        senders_mask = objects_mask
        weights = None
        if self.message_type in {'v1', 'relational'}:
            m_oht = compute_relational_message(receiver, senders, senders_mask,
                                               f=self.human_object_full_relation_mlp,
                                               g=self.human_object_pairwise_relation_mlp)
        else:  # v2 or non-relational
            m_oht = compute_non_relational_message(receiver, senders, senders_mask,
                                                   granularity=self.message_granularity,
                                                   message_fn=self.objects_to_human_message_mlp)
            if self.message_aggregation in {'mp', 'mean_pooling'}:
                num_real_senders = torch.sum(senders_mask, dim=1, keepdim=True)
                num_real_senders = torch.clamp(num_real_senders, min=1.0)
                m_oht = torch.sum(m_oht, dim=1) / num_real_senders
            else:  # attention or att
                if ho_dists is None:
                    att_weights = compute_attention_weights(receiver, senders, senders_mask,
                                                            attention_style=self.attention_style,
                                                            attention_fn=self.objects_to_human_message_att_mlp)
                else:
                    att_weights = compute_distance_based_attention_weights(ho_dists, senders_mask)
                weights = att_weights
                att_weights = torch.unsqueeze(att_weights, dim=-1)
                m_oht = torch.sum(att_weights * m_oht, dim=1)
        return m_oht, weights

    def _objects_to_human_segment_message(self, hx_hsh, hx_os, objects_mask, direction: str, ho_dists=None):
        """Compute a segment-level message from objects to human.

        Arguments:
            hx_hsh - A list containing the hidden states of human h up to time step t - 1.
            hx_os - A list containing the hidden states of all objects up to time step t - 1.
            objects_mask - A tensor of shape (batch_size, num_objects) containing information of whether an object
                is real or virtual.
            direction - Whether the message is for the forward or backward RNN.
            ho_dists - An optional tensor of shape (batch_size, num_objects) containing the distance between human h
                and all objects. Only meaningful in case message aggregation is attention, in which case the attention
                weights are dependent on the distance between human and objects.
        Returns:
            A tensor of shape (batch_size, hidden_size) containing the segment-level message from objects to human.
        """
        hidden_size = self.human_segment_rnn_fcell.hidden_size
        batch_size, dtype, device = objects_mask.size(0), objects_mask.dtype, objects_mask.device
        grab_state_fn = grab_last_state_or_zeros if direction == 'forward' else grab_first_state_or_zeros
        receiver = grab_state_fn(hx_hsh, batch_size, hidden_size, dtype, device, detach=False)
        senders = [grab_state_fn(hx_osk, batch_size, hidden_size, dtype, device, detach=False) for hx_osk in hx_os]
        senders = torch.stack(senders, dim=1)
        senders_mask = objects_mask
        weights = None
        if self.message_type in {'v1', 'relational'}:
            mg_oht = compute_relational_message(receiver, senders, senders_mask,
                                                f=self.human_object_segment_full_relation_mlp,
                                                g=self.human_object_segment_pairwise_relation_mlp)
        else:  # v2 or non-relational
            mg_oht = compute_non_relational_message(receiver, senders, senders_mask,
                                                    granularity=self.message_granularity,
                                                    message_fn=self.objects_to_human_segment_message_mlp)
            if self.message_aggregation in {'mp', 'mean_pooling'}:
                num_real_senders = torch.sum(senders_mask, dim=1, keepdim=True)
                num_real_senders = torch.clamp(num_real_senders, min=1.0)
                mg_oht = torch.sum(mg_oht, dim=1) / num_real_senders
            else:  # attention or att
                if ho_dists is None:
                    att_weights = compute_attention_weights(receiver, senders, senders_mask,
                                                            attention_style=self.attention_style,
                                                            attention_fn=self.objects_to_human_segment_message_att_mlp)
                else:
                    att_weights = compute_distance_based_attention_weights(ho_dists, senders_mask)
                weights = att_weights
                att_weights = torch.unsqueeze(att_weights, dim=-1)
                mg_oht = torch.sum(att_weights * mg_oht, dim=1)
        return mg_oht, weights

    def _objects_to_object_message(self, x_oft, h_oft, k, objects_mask, oo_dists=None):
        """Compute a message to an object from other objects.

        Arguments:
            x_oft - Tensor of shape (batch_size, num_objects, hidden_size) containing the frame-level input for the
                objects at time step t.
            h_oft - Tensor of shape (batch_size, num_objects, hidden_size) containing the frame-level hidden state for
                all objects at time step t.
            k - ID of the object that is receiving a message.
            objects_mask - A tensor of shape (batch_size, num_objects).
            oo_dists - An optional tensor of shape (batch_size, num_objects, num_objects) containing the distances
                between objects. Only meaningful in case message aggregation is attention, in which case the attention
                weights are dependent on the distance between the objects.
        Returns:
            A tensor of shape (batch_size, hidden_size) containing the message to object k from all other objects.
        """
        x_oftk = x_oft[:, k]
        h_oftk = h_oft[:, k]
        receiver = torch.cat([x_oftk, h_oftk], dim=-1)
        x_oft = torch.cat([x_oft[:, :k], x_oft[:, k + 1:]], dim=1)
        h_oft = torch.cat([h_oft[:, :k], h_oft[:, k + 1:]], dim=1)
        senders = torch.cat([x_oft, h_oft], dim=-1)
        senders_mask = torch.cat([objects_mask[:, :k], objects_mask[:, k + 1:]], dim=1)
        if self.message_type in {'v1', 'relational'}:
            m_ootk = compute_relational_message(receiver, senders, senders_mask,
                                                f=self.object_object_full_relation_mlp,
                                                g=self.object_object_pairwise_relation_mlp)
        else:
            m_ootk = compute_non_relational_message(receiver, senders, senders_mask,
                                                    granularity=self.message_granularity,
                                                    message_fn=self.objects_to_object_message_mlp)
            if self.message_aggregation in {'mp', 'mean_pooling'}:
                num_real_senders = torch.sum(senders_mask, dim=1, keepdim=True)
                num_real_senders = torch.clamp(num_real_senders, min=1.0)
                m_ootk = torch.sum(m_ootk, dim=1) / num_real_senders
            else:
                if oo_dists is None:
                    att_weights = compute_attention_weights(receiver, senders, senders_mask,
                                                            attention_style=self.attention_style,
                                                            attention_fn=self.objects_to_object_message_att_mlp)
                else:
                    oo_dists = oo_dists[:, k]
                    oo_dists = torch.cat([oo_dists[:, :k], oo_dists[:, k + 1:]], dim=-1)
                    att_weights = compute_distance_based_attention_weights(oo_dists, senders_mask)
                att_weights = torch.unsqueeze(att_weights, dim=-1)
                m_ootk = torch.sum(att_weights * m_ootk, dim=1)
        return m_ootk

    def _objects_to_object_segment_message(self, hx_os, objects_mask, k, direction: str, oo_dists=None):
        """Compute a segment-level message from objects to another object.

        Arguments:
            hx_os - A list containing the hidden states of all objects up to time step t - 1.
            objects_mask - A tensor of shape (batch_size, num_objects) containing information of whether an object is
                real or virtual.
            k - The object receiving the message.
            direction - Whether the message is for the 'forward' or 'backward' RNN.
            oo_dists - An optional tensor of shape (batch_size, num_objects, num_objects) containing the distances
                between objects. Only meaningful in case message aggregation is attention, in which case the attention
                weights are dependent on the distance between the objects.
        Returns:
            A tensor of shape (batch_size, hidden_size) containing the segment-level message to object k from all
            other real objects.
        """
        hidden_size = self.human_segment_rnn_fcell.hidden_size
        batch_size, dtype, device = objects_mask.size(0), objects_mask.dtype, objects_mask.device
        grab_state_fn = grab_last_state_or_zeros if direction == 'forward' else grab_first_state_or_zeros
        receiver = grab_state_fn(hx_os[k], batch_size, hidden_size, dtype, device, detach=False)
        senders = [grab_state_fn(hx_osm, batch_size, hidden_size, dtype, device, detach=False)
                   for hx_osm in hx_os[:k] + hx_os[k + 1:]]
        senders = torch.stack(senders, dim=1)
        senders_mask = torch.cat([objects_mask[:, :k], objects_mask[:, k + 1:]], dim=1)
        if self.message_type in {'v1', 'relational'}:
            mg_oot = compute_relational_message(receiver, senders, senders_mask,
                                                f=self.object_object_segment_full_relation_mlp,
                                                g=self.object_object_segment_pairwise_relation_mlp)
        else:
            mg_oot = compute_non_relational_message(receiver, senders, senders_mask,
                                                    granularity=self.message_granularity,
                                                    message_fn=self.objects_to_object_segment_message_mlp)
            if self.message_aggregation in {'mp', 'mean_pooling'}:
                num_real_senders = torch.sum(senders_mask, dim=1, keepdim=True)
                num_real_senders = torch.clamp(num_real_senders, min=1.0)
                mg_oot = torch.sum(mg_oot, dim=1) / num_real_senders
            else:
                if oo_dists is None:
                    att_weights = compute_attention_weights(receiver, senders, senders_mask,
                                                            attention_style=self.attention_style,
                                                            attention_fn=self.objects_to_object_segment_message_att_mlp)
                else:
                    oo_dists = oo_dists[:, k]
                    oo_dists = torch.cat([oo_dists[:, :k], oo_dists[:, k + 1:]], dim=-1)
                    att_weights = compute_distance_based_attention_weights(oo_dists, senders_mask)
                att_weights = torch.unsqueeze(att_weights, dim=-1)
                mg_oot = torch.sum(att_weights * mg_oot, dim=1)
        return mg_oot

    # geometry to object
    def _geometry_to_object_message(self, x_sft, x_oftk, h_sft, h_oftk, os_dists=None):
        """Compute human to an object message.

        For now, we assume that all humans are real.

        Arguments:
            x_sft - Tensor of shape (batch_size, num_humans, hidden_size) containing the frame-level input for the
                geometry at time step t.
            x_oftk - Tensor of shape (batch_size, hidden_size) containing the frame-level input for the k-th object
                at time step t.
            h_sft - Tensor of shape (batch_size, num_humans, hidden_size) containing the frame-level hidden state for
                the geometry at time step t.
            h_oftk - Tensor of shape (batch_size, hidden_size) containing frame-level hidden state of the k-th object
                at time step t.
            os_dists - An optional tensor of shape (batch_size, num_humans) containing the distances between the k-th
                object and the geometry. Only meaningful in case message aggregation is attention, in which case the
                attention weights are dependent on the distance between object and geometry.
        Returns:
            A tensor of shape (batch_size, hidden_size) containing the message from the geometry to the k-th object.
        """
        receiver = torch.cat([x_oftk, h_oftk], dim=-1)
        senders = torch.cat([x_sft, h_sft], dim=-1)
        senders_mask = torch.ones(senders.size()[:2], dtype=senders.dtype, device=senders.device)
        if self.message_type in {'v1', 'relational'}:
            hok_message = compute_relational_message(receiver, senders, senders_mask,
                                                     f=self.object_geometry_full_relation_mlp,
                                                     g=self.object_geometry_pairwise_relation_mlp)
        else:  # v2 or non-relational
            hok_message = compute_non_relational_message(receiver, senders, senders_mask,
                                                         granularity=self.message_granularity,
                                                         message_fn=self.geometry_to_object_message_mlp)
            if self.message_aggregation in {'mp', 'mean_pooling'}:
                num_real_senders = torch.sum(senders_mask, dim=1, keepdim=True)
                num_real_senders = torch.clamp(num_real_senders, min=1.0)
                hok_message = torch.sum(hok_message, dim=1) / num_real_senders
            else:
                if os_dists is None:
                    att_weights = compute_attention_weights(receiver, senders, senders_mask,
                                                            attention_style=self.attention_style,
                                                            attention_fn=self.geometry_to_object_message_att_mlp)
                else:
                    att_weights = compute_distance_based_attention_weights(os_dists, senders_mask)
                att_weights = torch.unsqueeze(att_weights, dim=-1)
                hok_message = torch.sum(att_weights * hok_message, dim=1)
        return hok_message

    # geometry to human
    def _geometry_to_human_message(self, x_sft, x_hftk, h_sft, h_hftk, hs_dists=None):
        """Compute human to an object message.

        For now, we assume that all humans are real.

        Arguments:
            x_sft - Tensor of shape (batch_size, num_humans, hidden_size) containing the frame-level input for the
                geometry at time step t.
            x_hftk - Tensor of shape (batch_size, hidden_size) containing the frame-level input for the k-th human
                at time step t.
            h_sft - Tensor of shape (batch_size, num_humans, hidden_size) containing the frame-level hidden state for
                the geometry at time step t.
            h_hftk - Tensor of shape (batch_size, hidden_size) containing frame-level hidden state of the k-th human
                at time step t.
            hs_dists - An optional tensor of shape (batch_size, num_humans) containing the distances between the k-th
                human and the geometry. Only meaningful in case message aggregation is attention, in which case the
                attention weights are dependent on the distance between human and geometry.
        Returns:
            A tensor of shape (batch_size, hidden_size) containing the message from the geometry to the k-th human.
        """
        receiver = torch.cat([x_hftk, h_hftk], dim=-1)
        senders = torch.cat([x_sft, h_sft], dim=-1)
        senders_mask = torch.ones(senders.size()[:2], dtype=senders.dtype, device=senders.device)
        if self.message_type in {'v1', 'relational'}:
            hok_message = compute_relational_message(receiver, senders, senders_mask,
                                                     f=self.human_geometry_full_relation_mlp,
                                                     g=self.human_geometry_pairwise_relation_mlp)
        else:  # v2 or non-relational
            hok_message = compute_non_relational_message(receiver, senders, senders_mask,
                                                         granularity=self.message_granularity,
                                                         message_fn=self.geometry_to_human_message_mlp)
            if self.message_aggregation in {'mp', 'mean_pooling'}:
                num_real_senders = torch.sum(senders_mask, dim=1, keepdim=True)
                num_real_senders = torch.clamp(num_real_senders, min=1.0)
                hok_message = torch.sum(hok_message, dim=1) / num_real_senders
            else:
                if hs_dists is None:
                    att_weights = compute_attention_weights(receiver, senders, senders_mask,
                                                            attention_style=self.attention_style,
                                                            attention_fn=self.geometry_to_human_message_att_mlp)
                else:
                    att_weights = compute_distance_based_attention_weights(hs_dists, senders_mask)
                att_weights = torch.unsqueeze(att_weights, dim=-1)
                hok_message = torch.sum(att_weights * hok_message, dim=1)
        return hok_message

    def _update_human_segment(self, x_hfth, h_hfth, m_hhth, m_ohth, m_shth, x_tt):
        """Operation to update current human segment-level discrete state.

        Arguments:
            x_hfth - Tensor of shape (batch_size, hidden_size) containing the frame-level input for human h
                at time step t.
            h_hfth - Tensor of shape (batch_size, hidden_size) containing the frame-level hidden state of human h at
                time step t.
            m_hhth - Tensor of shape (batch_size, hidden_size) containing a message from the other humans to human h
                at time step t.
            m_ohth - Tensor of shape (batch_size, hidden_size) containing a message from the objects to human h
                at time step t.
            x_tt - Tensor of shape (batch_size, hidden_size) containing the time embedding.
        Returns:
            Two tensors of shape (batch_size, 1) containing the binary decision of whether to update the segment-level
            RNN or not. The first tensor contains the hard decision whereas the second contains the soft decision.
        """
        update_human_segment_input = cat_valid_tensors([x_hfth, h_hfth, m_hhth, m_ohth, m_shth, x_tt], dim=-1)
        u_hsts = self.update_human_segment_mlp(update_human_segment_input)
        u_hst, u_hsts = discrete_estimator(u_hsts, strategy=self.discrete_optimization_strategy,
                                           threshold=self.update_segment_threshold)
        return u_hst, u_hsts

    def _update_object_segment(self, x_oftk, h_oftk, m_hotk, m_ootk, m_sotk, u_hst, u_hsts, x_tt):
        """Operation to update current human segment-level discrete state.

        Arguments:
            x_oftk - Tensor of shape (batch_size, hidden_size) containing the frame-level input for the k-th object
                at time step t.
            h_oftk - Tensor of shape (batch_size, hidden_size) containing frame-level hidden state of the k-th object
                at time step t.
            m_hotk - Tensor of shape (batch_size, hidden_size) containing a message from the humans to the k-th object
                at time step t.
            m_ootk - Tensor of shape (batch_size, hidden_size) containing a message from all other objects to object
                k at time step t.
            u_hst - Tensor of shape (batch_size, 1) containing the hard binary decision of whether to update the human
                segment-level RNN at time step t. Note that this only makes sense if we only have a single human in the
                scene; if there are multiple humans in the scene this should be None.
            u_hsts - Tensor of shape (batch_size, 1) containing the soft binary decision of whether to update the human
                segment-level RNN at time step t. Similarly to u_hst, it should be None in case of multiple humans in
                the scene.
            x_tt - Tensor of shape (batch_size, hidden_size) containing the time embedding.
        Returns:
            Two tensors of shape (batch_size, 1) containing the binary decision of whether to update the segment-level
            RNN or not. The first tensor contains the hard decision whereas the second contains the soft decision.
        """
        if self.object_segment_update_strategy in {'same_as_human', 'sah'} and u_hst is not None and u_hsts is not None:
            u_ostk = u_hst
            u_ostks = u_hsts
        else:  # independent
            update_object_segment_input = cat_valid_tensors([x_oftk, h_oftk, m_hotk, m_ootk, m_sotk, x_tt], dim=-1)
            u_ostks = self.update_object_segment_mlp(update_object_segment_input)
            u_ostk, u_ostks = discrete_estimator(u_ostks, strategy=self.discrete_optimization_strategy,
                                                 threshold=self.update_segment_threshold)
            if self.object_segment_update_strategy in {'conditional_on_human', 'coh'} and u_hst is not None:
                u_ostk = u_ostk * u_hst
        return u_ostk, u_ostks

    def _bidirectional_step(self, x_st, u_st, hx_s, entity: str, direction: str):
        """Perform a single forward or backward step for the segment-level RNN.

        Arguments:
            x_st - Tensor of shape (batch_size, segment_input_size) containing the input for the segment-level RNN
                at time step t.
            u_st - Tensor of shape (batch_size, 1) containing the hard decision of whether the segment-level RNN
                updates at time step t.
            hx_s - A list containing the segment-level hidden state up to time step t - 1.
            entity - Either 'human' or 'object'.
            direction - Either 'forward' or 'backward'.
        Returns:
            A tensor of shape (batch_size, hidden_size) containing the hidden state of the segment-level RNN at time
            step t.
        """
        batch_size, hidden_size = x_st.size(0), self.human_segment_rnn_fcell.hidden_size
        dtype, device = x_st.dtype, x_st.device
        if direction == 'forward':
            h_st = grab_last_state_or_zeros(hx_s, batch_size, hidden_size, dtype, device, detach=False)
            if entity == 'human':
                h_st = u_st * self.human_segment_rnn_fcell(x_st, h_st) + (1.0 - u_st) * h_st
            else:
                h_st = u_st * self.object_segment_rnn_fcell(x_st, h_st) + (1.0 - u_st) * h_st
        else:
            h_st = grab_first_state_or_zeros(hx_s, batch_size, hidden_size, dtype, device, detach=False)
            if entity == 'human':
                h_st = u_st * self.human_segment_rnn_bcell(x_st, h_st) + (1.0 - u_st) * h_st
            else:
                h_st = u_st * self.object_segment_rnn_bcell(x_st, h_st) + (1.0 - u_st) * h_st
        return h_st


def reorder_hidden_states(hx_s, ux_s):
    """Make hidden states of end frames to be the hidden states of previous frames.

    Arguments:
        hx_s - A tensor of shape (batch_size, num_steps, hidden_size) containing the segment-level hidden states.
        ux_s - A binary tensor of shape (batch_size, num_steps) specifying whether the a frame is an end frame or
            not.
    Returns:
        The reordered version of hx_s (same shape).
    """
    batch_size = hx_s.size(0)
    hx_s = [list(torch.unbind(hx_m)) for hx_m in torch.unbind(hx_s)]
    for m in range(batch_size):
        u_sm = ux_s[m]
        end_frames = [-1] + torch.nonzero(u_sm, as_tuple=True)[0].tolist()
        for start_frame, end_frame in zip(end_frames[:-1], end_frames[1:]):
            for t in range(start_frame + 1, end_frame):
                hx_s[m][t] = hx_s[m][end_frame]
    hx_s = torch.stack([torch.stack(hx_m, dim=0) for hx_m in hx_s], dim=0)
    return hx_s


def select_model(model_name: str):
    model_name_to_class_definition = {
        'bimanual_baseline': BimanualBaseline,
        'cad120_baseline': CAD120Baseline,
        '2G-GCN': TGGCN,
    }
    return model_name_to_class_definition[model_name]


def grab_last_state_or_zeros(hx, batch_size, hidden_size, dtype, device, detach=False):
    try:
        hx_t = hx[-1]
    except IndexError:
        hx_t = torch.zeros([batch_size, hidden_size], dtype=dtype, device=device)
    else:
        if detach:
            hx_t = hx_t.detach()
    return hx_t


def grab_first_state_or_zeros(hx, batch_size, hidden_size, dtype, device, detach=False):
    try:
        hx_t = hx[0]
    except IndexError:
        hx_t = torch.zeros([batch_size, hidden_size], dtype=dtype, device=device)
    else:
        if detach:
            hx_t = hx_t.detach()
    return hx_t


def discrete_estimator(x, strategy: str = 'straight-through', threshold: float = 0.5):
    if strategy in {'straight-through', 'st'}:
        return straight_through_estimator(x, threshold), x
    elif strategy in {'gumbel-sigmoid', 'gs'}:
        z, x = straight_through_gumbel_sigmoid(x, threshold=threshold)
        return z, x
    else:
        raise ValueError(f'strategy must be either straight-through or gumbel-sigmoid, not {strategy}.')


def load_model_weights(model_dir: str):
    checkpoint_file = os.path.join(model_dir, os.path.basename(model_dir) + '.tar')
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint['model_state_dict']
    return state_dict


def filter_soft_decisions(ux_s: list, update_threshold: float = 0.5):
    """Filter discrete updates for their local maximum.

    Arguments:
        ux_s - A list containing tensors of shape (batch_size, 1) containing the soft decisions for each time step.
    Returns:
        A list containing the filtered hard decisions.
    """
    num_steps = len(ux_s)
    ux_h = []
    for t in range(num_steps):
        u_s_t = ux_s[t]
        if t:
            u_s_tm1 = ux_s[t - 1]
        else:
            u_s_tm1 = torch.zeros_like(u_s_t)
        try:
            u_s_tp1 = ux_s[t + 1]
        except IndexError:
            u_s_tp1 = torch.zeros_like(u_s_t)
        condition = u_s_t > u_s_tm1
        condition = torch.min(condition, u_s_t > u_s_tp1)
        condition = torch.min(condition, u_s_t >= update_threshold)
        u_h_t = (u_s_t >= update_threshold).float()
        u_h_t = (u_h_t - u_s_t).detach() + u_s_t
        u_h_t = torch.where(condition, u_h_t, torch.clamp(u_h_t, max=0.0))
        ux_h.append(u_h_t)
    return ux_h


def compute_relational_message(receiver, senders, senders_mask, f, g):
    """General function to compute a relational message between many senders and a single receiver.

    Arguments:
        receiver - A tensor of shape (batch_size, receiver_feature_size) containing the features representing the
            receiver.
        senders - A tensor of shape (batch_size, num_senders, sender_feature_size) containing the features representing
            the senders.
        senders_mask - A binary tensor of shape (batch_size, num_senders) specifying whether a sender is real or
            virtual.
        f - A PyTorch model that computes the full relation.
        g - A PyTorch model that computes the pairwise relation between the receiver and the senders.
    Returns:
        A tensor of shape (batch_size, relational_message_size) containing a message from the senders to the receiver.
    """
    sr_relations = []
    num_senders = senders.size(1)
    for s in range(num_senders):
        sender = senders[:, s]
        x_sr_pairwise = torch.cat([receiver, sender], dim=-1)
        sr_relation = g(x_sr_pairwise) * senders_mask[:, s:s + 1]
        sr_relations.append(sr_relation)
    m_sr = f(sum(sr_relations))
    return m_sr


def compute_non_relational_message(receiver, senders, senders_mask, granularity, message_fn):
    """General function to compute a non-relational message between many senders a single receiver.

    Arguments:
        receiver - A tensor of shape (batch_size, receiver_feature_size) containing the features representing the
            receiver.
        senders - A tensor of shape (batch_size, num_senders, sender_feature_size) containing the features representing
            the senders.
        senders_mask - A binary tensor of shape (batch_size, num_senders) specifying whether a sender is real or
            virtual.
        granularity - Whether the message from the sender solely depends on the sender ('v1' or 'generic') or also
            depends on the receiver ('v2' or 'specific').
        message_fn - PyTorch function to compute a message from the sender.
    Returns:
        A tensor of shape (batch_size, num_senders, message_feature_size) containing the messages of each sender.
    """
    mx_sr = []
    num_senders = senders.size(1)
    for s in range(num_senders):
        sender = senders[:, s]
        if granularity in {'v2', 'specific'}:
            sender = torch.cat([receiver, sender], dim=-1)
        m_sr = message_fn(sender) * senders_mask[:, s:s + 1]
        mx_sr.append(m_sr)
    mx_sr = torch.stack(mx_sr, dim=1)
    return mx_sr


def compute_attention_weights(query, keys, keys_mask, attention_style, attention_fn=None):
    """Compute attention weights.

    We assume query_size and key_size are the same.

    Arguments:
        query - A tensor of shape (batch_size, query_size) containing the query features.
        keys - A tensor of shape (batch_size, num_keys, key_size) containing the key features.
        keys_mask - A binary tensor of shape (batch_size, num_keys) specifying whether a key is real or virtual.
        attention_style - How to perform the attention.
        attention_fn - A PyTorch function to compute the similarity between the query and a key.
    Returns:
        A tensor of shape (batch_size, num_keys) containing the attention weights.
    """
    att_weights = []
    num_senders = keys.size(1)
    for s in range(num_senders):
        key = keys[:, s]
        if attention_style in {'v1', 'concat'}:
            att_fn_input = torch.cat([query, key], dim=-1)
            att_weight = attention_fn(att_fn_input)
        elif attention_style in {'v2', 'dot-product', 'v3', 'scaled_dot-product'}:
            att_weight = torch.sum(query * key, dim=-1, keepdim=True)
            if attention_style in {'v3', 'scaled_dot-product'}:
                att_weight = att_weight / math.sqrt(key.size(-1))
        else:  # v4 or general
            att_weight = torch.relu(attention_fn(query, key))
        att_weights.append(att_weight)
    att_weights = torch.cat(att_weights, dim=-1)
    neg_inf_values = torch.full_like(att_weights, fill_value=float('-inf'))
    att_weights = torch.where(keys_mask.bool(), att_weights, neg_inf_values)
    att_weights = torch.nn.functional.softmax(att_weights, dim=1)
    att_weights = torch.where(torch.isnan(att_weights), torch.zeros_like(att_weights), att_weights)
    return att_weights


def compute_distance_based_attention_weights(distances, senders_mask):
    """Compute attention weights based on distances between entities.

    Arguments:
        distances - A tensor of shape (batch_size, num_senders) containing the distances between each sender and the
            receiver.
        senders_mask - A binary tensor of shape (batch_size, num_senders) specifying whether a sender is real or
            virtual.
    Returns:
        A tensor of shape (batch_size, num_senders) containing the attention weight of each sender.
    """
    distance_mask = distances.bool()
    negative_infinity_values = torch.full_like(distances, fill_value=float('-inf'))
    distances = 1 / (distances + 1e-7)
    distances = torch.where(senders_mask.bool(), distances, negative_infinity_values)
    distances = torch.where(distance_mask.bool(), distances, negative_infinity_values)
    att_weights = torch.nn.functional.softmax(distances, dim=-1)
    att_weights = torch.where(torch.isnan(att_weights), torch.zeros_like(att_weights), att_weights)
    return att_weights


def make_periodic_embedding(x, hidden_size: int):
    """Assemble a periodic embedding tensor.

    Arguments:
        x - A tensor of shape (*, 1).
        hidden_size - Desired hidden size for periodic embedding. Must be a multiple of two.
    Returns:
        A tensor of shape (*, hidden_size) containing the periodic embedding of the input tensor.
    """
    assert (hidden_size % 2) == 0, 'hidden_size must be even.'
    w = torch.tensor([1e4], dtype=x.dtype, device=x.device)
    exponent = torch.linspace(0, 1, hidden_size // 2, dtype=x.dtype, device=x.device)
    w = w ** exponent
    sines = torch.sin(x / w)
    cosines = torch.cos(x / w)
    output = torch.cat([sines, cosines], dim=-1)
    return output
