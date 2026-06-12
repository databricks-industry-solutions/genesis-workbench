# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Copyright (c) 2021 Tencent AI Lab.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
The KERMT models for pretraining, finetuning and fingerprint generating.
"""
from argparse import Namespace
from typing import List, Dict, Callable, Tuple
import math

import numpy as np
import torch
from torch import nn as nn
from torch.cuda import nvtx

from kermt.data import get_atom_fdim, get_bond_fdim
from kermt.model.layers import Readout, GTransEncoder, RoPETransformerDecoderLayer, PositionalEncoding
from kermt.util.loss_utils import normalize_loss_gradient
from kermt.util.nn_utils import get_activation_function


# ============================================================================
# Shared helper functions for CMIM loss computation
# ============================================================================

def compute_batchwise_cosine_sim(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cosine similarity between latent vectors in a batch.
    Shared by KermtCMIMTask and KermtHybridTask.
    
    Args:
        z: latent vectors [batch_size, latent_dim]
        
    Returns:
        sim_to_pos: similarity to self [batch_size] (always 1.0)
        sim_to_neg: similarity to other samples [batch_size, batch_size-1]
    """
    b = z.shape[0]
    z = z.view(b, -1)  # Flatten if needed
    
    # Normalize to unit vectors
    z_norm = torch.nn.functional.normalize(z, dim=1)
    
    # Similarity to positive (self) - should be 1.0
    sim_to_pos = (z_norm * z_norm).sum(dim=1)  # [b]
    
    # Similarity matrix (all pairs)
    sim_matrix = z_norm @ z_norm.T  # [b, b]
    
    # Mask out diagonal and extract negative pairs
    mask = ~torch.eye(b, dtype=bool, device=z.device)
    sim_to_neg = sim_matrix[mask].view(b, b - 1)  # [b, b-1]
    
    return sim_to_pos, sim_to_neg


def compute_cmim_loss(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    z_latent: torch.Tensor,
    contrastive_temperature: float,
    normalize_gradient: bool = False,
    normalize_loss: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute CMIM loss (without reconstruction component).
    Shared by KermtCMIMTask and KermtHybridTask.
    
    Args:
        mean: mean of latent distribution [batch_size, latent_dim]
        log_scale: log scale of latent distribution [batch_size, latent_dim]
        z_latent: sampled latent vectors [batch_size, latent_dim]
        contrastive_temperature: temperature parameter for contrastive loss
        normalize_gradient: whether to normalize gradients by latent dimensionality
        normalize_loss: whether to normalize loss values by latent dimensionality
        
    Returns:
        dictionary containing loss components:
            - cmim_loss: the CMIM loss [batch_size]
            - log_p_k1_given_zx: contrastive log probability [batch_size]
            - log_q_z_given_x: encoder log probability [batch_size]
            - log_P_z: prior log probability [batch_size]
    """
    b = mean.shape[0]
    
    # Compute normalization coefficient for latent space
    # Following MIM-playground: 1 / product of all non-batch dimensions
    z_grad_coeff = 1.0 / torch.prod(torch.tensor(z_latent.shape[1:], dtype=torch.float32)).item()
    
    # q(z|x) - the encoder distribution
    q_z_given_x = torch.distributions.Normal(
        loc=mean,
        scale=torch.exp(log_scale),
    )
    
    # P(z) - the prior (standard normal)
    P_z = torch.distributions.Normal(
        loc=torch.zeros_like(mean),
        scale=torch.ones_like(log_scale),
    )
    
    # Contrastive loss component: log p(k=1|z,x)
    if b > 1:
        # Compute cosine similarity between z and itself, and z and other samples
        z_sim_to_pos, z_sim_to_neg = compute_batchwise_cosine_sim(z_latent)  # [b], [b, b-1]
        
        # Scale by temperature to get logits
        pos_logits = z_sim_to_pos / contrastive_temperature  # [b]
        neg_logits = z_sim_to_neg / contrastive_temperature  # [b, b-1]
        
        # Numerically stable log-softmax for contrastive loss
        # log p(k=1|z,x) = pos_logits - log(exp(pos_logits) + sum(exp(neg_logits)))
        log_p_k1_given_zx = (
            pos_logits - 
            torch.logsumexp(
                torch.cat([pos_logits.unsqueeze(1), neg_logits - math.log(b - 1)], dim=1),
                dim=1
            )
        )  # [b]
    else:
        # Batch size 1: skip contrastive loss
        log_p_k1_given_zx = torch.zeros(b, device=z_latent.device)
    
    # KL divergence components - apply gradient normalization if enabled
    log_q_z_given_x = normalize_loss_gradient(
        q_z_given_x.log_prob(z_latent).sum(-1),  # [b, latent_dim] => [b]
        z_grad_coeff,
        normalize_gradient=normalize_gradient,
        normalize_loss=normalize_loss,
    )
    log_P_z = normalize_loss_gradient(
        P_z.log_prob(z_latent).sum(-1),  # [b, latent_dim] => [b]
        z_grad_coeff,
        normalize_gradient=normalize_gradient,
        normalize_loss=normalize_loss,
    )
    
    # CMIM loss: -[log p(k=1|z,x) + 0.5 * (log q(z|x) + log P(z))]
    cmim_loss = -(log_p_k1_given_zx + 0.5 * (log_q_z_given_x + log_P_z))  # [b]
    
    return {
        "cmim_loss": cmim_loss,
        "log_p_k1_given_zx": log_p_k1_given_zx,
        "log_q_z_given_x": log_q_z_given_x,
        "log_P_z": log_P_z,
    }


class KERMTEmbedding(nn.Module):
    """
    The KERMT Embedding class. It contains the GTransEncoder.
    This GTransEncoder can be replaced by any validate encoders.
    """

    def __init__(self, args: Namespace):
        """
        Initialize the KERMTEmbedding class.
        :param args:
        """
        super(KERMTEmbedding, self).__init__()
        self.embedding_output_type = args.embedding_output_type
        edge_dim = get_bond_fdim() + get_atom_fdim()
        node_dim = get_atom_fdim()
        if not hasattr(args, "backbone"):
            print("No backbone specified in args, use gtrans backbone.")
            args.backbone = "gtrans"
        if args.backbone == "gtrans" or args.backbone == "dualtrans":
            # dualtrans is the old name.
            self.encoders = GTransEncoder(args,
                                          hidden_size=args.hidden_size,
                                          edge_fdim=edge_dim,
                                          node_fdim=node_dim,
                                          dropout=args.dropout,
                                          activation=args.activation,
                                          num_mt_block=args.num_mt_block,
                                          num_attn_head=args.num_attn_head,
                                          embedding_output_type=self.embedding_output_type,
                                          bias=args.bias,
                                          cuda=args.cuda)

    def forward(self, graph_batch: List) -> Dict:
        """
        The forward function takes graph_batch as input and output a dict. The content of the dict is decided by
        self.embedding_output_type.

        :param graph_batch: the input graph batch generated by MolCollator.
        :return: a dict containing the embedding results.
        """
        output = self.encoders(graph_batch)
        if self.embedding_output_type == 'atom':
            return {"atom_from_atom": output[0], "atom_from_bond": output[1],
                    "bond_from_atom": None, "bond_from_bond": None}  # atom_from_atom, atom_from_bond
        elif self.embedding_output_type == 'bond':
            return {"atom_from_atom": None, "atom_from_bond": None,
                    "bond_from_atom": output[0], "bond_from_bond": output[1]}  # bond_from_atom, bond_from_bond
        elif self.embedding_output_type == "both":
            return {"atom_from_atom": output[0][0], "bond_from_atom": output[0][1],
                    "atom_from_bond": output[1][0], "bond_from_bond": output[1][1]}


class KERMTLatentDistribution(nn.Module):
    """
    Extract latent distribution parameters (mean and log_scale) from KERMT embeddings.
    This is used for computing CMIM loss.
    
    Example usage:
        # Initialize
        latent_dist = KERMTLatentDistribution(args, latent_dim=512)
        
        # In training loop
        mean, log_scale = latent_dist(batch)  # Get distribution parameters
        z_latent = latent_dist.sample(batch)  # Sample from distribution
        
        # Or get both at once
        z_latent, mean, log_scale = latent_dist.sample(batch, return_params=True)
    """

    def __init__(self, args: Namespace, kermt: KERMTEmbedding = None, 
                 latent_dim: int = None, min_log_scale: float = -6.0):
        """
        Initialize the KERMTLatentDistribution class.
        
        :param args: the arguments containing model configuration
        :param kermt: optional pre-initialized KERMTEmbedding instance. If None, creates new one.
        :param latent_dim: dimension of the latent space. If None, defaults to args.hidden_size.
        :param min_log_scale: minimum value for log_scale to prevent numerical instability.
        """
        super(KERMTLatentDistribution, self).__init__()
        
        self.kermt = kermt if kermt is not None else KERMTEmbedding(args)
        self.embedding_output_type = args.embedding_output_type
        self.min_log_scale = min_log_scale
        
        # Determine dimensions
        self.hidden_size = args.hidden_size  # Encoder hidden size
        self.latent_dim = latent_dim if latent_dim is not None else args.latent_dim
        
        # Create readout layer for aggregating atom/bond embeddings to molecule level
        # Consistent with KermtFinetuneTask
        if args.self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=self.hidden_size,
                                   attn_hidden=args.attn_hidden,
                                   attn_out=args.attn_out)
        else:
            self.readout = Readout(rtype="mean", hidden_size=self.hidden_size)
        
        # Linear layer to project averaged molecule embedding to mean and log_scale
        # Output is 2 * latent_dim (half for mean, half for log_scale)
        self.fc_mean_logscale = nn.Linear(self.hidden_size, 2 * self.latent_dim)
        
    def forward(self, batch: List) -> tuple:
        """
        Forward pass to extract mean and log_scale from KERMT embeddings.
        
        :param batch: graph batch input (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)
        :return: tuple of (mean, log_scale) where both are tensors of shape [batch_size, latent_dim]
        """
        # Extract scope from batch
        _, _, _, _, _, a_scope, b_scope, _ = batch
        
        # Get embeddings from KERMT
        embeddings = self.kermt(batch)
        
        # Select which embeddings to use based on embedding_output_type
        if self.embedding_output_type == 'atom':
            # Use only atom embeddings (2 branches)
            mol_emb_1 = self.readout(embeddings["atom_from_atom"], a_scope)  # [batch_size, hidden_size]
            mol_emb_2 = self.readout(embeddings["atom_from_bond"], a_scope)  # [batch_size, hidden_size]
            mol_emb = (mol_emb_1 + mol_emb_2) / 2  # [batch_size, hidden_size]
            
        elif self.embedding_output_type == 'bond':
            # Use only bond embeddings (2 branches)
            mol_emb_1 = self.readout(embeddings["bond_from_atom"], b_scope)  # [batch_size, hidden_size]
            mol_emb_2 = self.readout(embeddings["bond_from_bond"], b_scope)  # [batch_size, hidden_size]
            mol_emb = (mol_emb_1 + mol_emb_2) / 2  # [batch_size, hidden_size]
            
        elif self.embedding_output_type == 'both':
            # Use all four embeddings (4 branches)
            mol_emb_atom_from_atom = self.readout(embeddings["atom_from_atom"], a_scope)  # [batch_size, hidden_size]
            mol_emb_atom_from_bond = self.readout(embeddings["atom_from_bond"], a_scope)  # [batch_size, hidden_size]
            mol_emb_bond_from_atom = self.readout(embeddings["bond_from_atom"], b_scope)  # [batch_size, hidden_size]
            mol_emb_bond_from_bond = self.readout(embeddings["bond_from_bond"], b_scope)  # [batch_size, hidden_size]
            mol_emb = (mol_emb_atom_from_atom + mol_emb_atom_from_bond + 
                      mol_emb_bond_from_atom + mol_emb_bond_from_bond) / 4  # [batch_size, hidden_size]
        else:
            raise ValueError(f"Unknown embedding_output_type: {self.embedding_output_type}")
        
        # Project to mean and log_scale
        params = self.fc_mean_logscale(mol_emb)  # [batch_size, 2 * latent_dim]
        mean, log_scale = params.chunk(2, dim=-1)  # Each: [batch_size, latent_dim]
        
        # Clamp log_scale to prevent numerical instability
        log_scale = torch.clamp(log_scale, min=self.min_log_scale)
        
        return mean, log_scale
    
    def sample(self, batch: List, return_params: bool = False):
        """
        Sample from the latent distribution using the reparameterization trick.
        
        :param batch: graph batch input
        :param return_params: if True, also returns mean and log_scale
        :return: z_latent [batch_size, latent_dim], optionally (mean, log_scale)
        """
        mean, log_scale = self.forward(batch)
        
        # Reparameterization trick: z = mean + std * epsilon, where epsilon ~ N(0, 1)
        std = torch.exp(log_scale)
        epsilon = torch.randn_like(std)
        z_latent = mean + std * epsilon
        
        if return_params:
            return z_latent, mean, log_scale
        return z_latent


class AtomVocabPrediction(nn.Module):
    """
    The atom-wise vocabulary prediction task. The atom vocabulary is constructed by the context.
    """
    def __init__(self, args, vocab_size, hidden_size=None):
        """
        :param args: the argument.
        :param vocab_size: the size of atom vocabulary.
        """
        super(AtomVocabPrediction, self).__init__()
        if not hidden_size:
            hidden_size = args.hidden_size
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, embeddings):
        """
        If embeddings is None: do not go through forward pass.
        :param embeddings: the atom embeddings, num_atom X fea_dim.
        :return: the prediction for each atom, num_atom X vocab_size.
        """
        if embeddings is None:
            return None
        return self.logsoftmax(self.linear(embeddings))


class BondVocabPrediction(nn.Module):
    """
    The bond-wise vocabulary prediction task. The bond vocabulary is constructed by the context.
    """
    def __init__(self, args, vocab_size, hidden_size=None):
        """
        Might need to use different architecture for bond vocab prediction.
        :param args:
        :param vocab_size: size of bond vocab.
        :param hidden_size: hidden size
        """
        super(BondVocabPrediction, self).__init__()
        if not hidden_size:
            hidden_size = args.hidden_size
        self.linear = nn.Linear(hidden_size, vocab_size)

        # ad-hoc here
        # If TWO_FC_4_BOND_VOCAB, we will use two distinct fc layer to deal with the bond and rev bond.
        self.TWO_FC_4_BOND_VOCAB = True
        if self.TWO_FC_4_BOND_VOCAB:
            self.linear_rev = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, embeddings):
        """
        If embeddings is None: do not go through forward pass.
        :param embeddings: the atom embeddings, num_bond X fea_dim.
        :return: the prediction for each atom, num_bond X vocab_size.
        """
        if embeddings is None:
            return None
        nm_bonds = embeddings.shape[0]  # must be an odd number
        # The bond and rev bond have odd and even ids respectively. See definition in molgraph.
        ids1 = [0] + list(range(1, nm_bonds, 2))
        ids2 = list(range(0, nm_bonds, 2))
        if self.TWO_FC_4_BOND_VOCAB:
            logits = self.linear(embeddings[ids1]) + self.linear_rev(embeddings[ids2])
        else:
            logits = self.linear(embeddings[ids1] + embeddings[ids2])

        return self.logsoftmax(logits)


class FunctionalGroupPrediction(nn.Module):
    """
    The functional group (semantic motifs) prediction task. This is a graph-level task.
    """
    def __init__(self, args, fg_size):
        """
        :param args: The arguments.
        :param fg_size: The size of semantic motifs.
        """
        super(FunctionalGroupPrediction, self).__init__()
        first_linear_dim = args.hidden_size
        hidden_size = args.hidden_size

        # In order to retain maximal information in the encoder, we use a simple readout function here.
        self.readout = Readout(rtype="mean", hidden_size=hidden_size)
        # We have four branches here. But the input with less than four branch is OK.
        # Since we use BCEWithLogitsLoss as the loss function, we only need to output logits here.
        self.linear_atom_from_atom = nn.Linear(first_linear_dim, fg_size)
        self.linear_atom_from_bond = nn.Linear(first_linear_dim, fg_size)
        self.linear_bond_from_atom = nn.Linear(first_linear_dim, fg_size)
        self.linear_bond_from_bond = nn.Linear(first_linear_dim, fg_size)

    def forward(self, embeddings: Dict, ascope: List, bscope: List) -> Dict:
        """
        The forward function of semantic motif prediction. It takes the node/bond embeddings, and the corresponding
        atom/bond scope as input and produce the prediction logits for different branches.
        :param embeddings: The input embeddings are organized as dict. The output of KERMTEmbedding.
        :param ascope: The scope for bonds. Please refer BatchMolGraph for more details.
        :param bscope: The scope for aotms. Please refer BatchMolGraph for more details.
        :return: a dict contains the predicted logits.
        """

        preds_atom_from_atom, preds_atom_from_bond, preds_bond_from_atom, preds_bond_from_bond = \
            None, None, None, None

        if embeddings["bond_from_atom"] is not None:
            preds_bond_from_atom = self.linear_bond_from_atom(self.readout(embeddings["bond_from_atom"], bscope))
        if embeddings["bond_from_bond"] is not None:
            preds_bond_from_bond = self.linear_bond_from_bond(self.readout(embeddings["bond_from_bond"], bscope))

        if embeddings["atom_from_atom"] is not None:
            preds_atom_from_atom = self.linear_atom_from_atom(self.readout(embeddings["atom_from_atom"], ascope))
        if embeddings["atom_from_bond"] is not None:
            preds_atom_from_bond = self.linear_atom_from_bond(self.readout(embeddings["atom_from_bond"], ascope))

        return {"atom_from_atom": preds_atom_from_atom, "atom_from_bond": preds_atom_from_bond,
                "bond_from_atom": preds_bond_from_atom, "bond_from_bond": preds_bond_from_bond}


class SMILESTransformerDecoder(nn.Module):
    """
    Transformer decoder for SMILES reconstruction with configurable positional encoding.
    Takes molecular latent representation and autoregressively generates SMILES.
    
    Supports both RoPE (Rotary Position Embedding) and classic sinusoidal positional encoding.
    Optionally supports G1 gating (SDPA output gating) for improved attention performance.
    Gating can be independently enabled for self-attention and cross-attention.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int = 512,
                 num_layers: int = 3,
                 num_attention_heads: int = 12,
                 ffn_hidden_size: int = 2048,  # Typically 4 * hidden_size
                 dropout: float = 0.1,
                 max_seq_len: int = 512,
                 pad_token_id: int = 0,
                 positional_encoding: str = 'rope',
                 gate_self_attn: bool = False,
                 gate_cross_attn: bool = False):
        """
        Initialize the SMILES transformer decoder.
        
        :param vocab_size: size of SMILES vocabulary
        :param hidden_size: dimension of hidden representations
        :param num_layers: number of transformer decoder layers
        :param num_attention_heads: number of attention heads
        :param ffn_hidden_size: dimension of feedforward network
        :param dropout: dropout rate
        :param max_seq_len: maximum sequence length
        :param pad_token_id: ID of padding token in vocabulary
        :param positional_encoding: type of positional encoding ('rope' or 'sinusoidal')
        :param gate_self_attn: whether to apply G1 gating to self-attention (only for 'rope' mode)
        :param gate_cross_attn: whether to apply G1 gating to cross-attention (only for 'rope' mode)
        """
        super(SMILESTransformerDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.positional_encoding = positional_encoding
        self.gate_self_attn = gate_self_attn
        self.gate_cross_attn = gate_cross_attn
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        
        # Positional encoding (type-specific)
        if positional_encoding == 'rope':
            # RoPE: no explicit positional encoding module (applied within attention)
            self.pos_encoder = None
            self.input_dropout = nn.Dropout(dropout)
            
            # Transformer decoder layers with RoPE and optional G1 gating
            self.decoder_layers = nn.ModuleList([
                RoPETransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=ffn_hidden_size,
                    dropout=dropout,
                    activation='relu',
                    max_seq_len=max_seq_len,
                    gate_self_attn=gate_self_attn,
                    gate_cross_attn=gate_cross_attn
                )
                for _ in range(num_layers)
            ])
            self.transformer_decoder = None
            
        elif positional_encoding == 'sinusoidal':
            # Classic sinusoidal positional encoding
            self.pos_encoder = PositionalEncoding(hidden_size, dropout, max_seq_len)
            self.input_dropout = None
            
            # Note: G1 gating is not supported with sinusoidal encoding (uses standard PyTorch layers)
            if gate_self_attn or gate_cross_attn:
                import warnings
                warnings.warn("gate_self_attn/gate_cross_attn are only supported with "
                            "positional_encoding='rope'. Ignoring for sinusoidal mode.")
            
            # Standard PyTorch transformer decoder layers
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=ffn_hidden_size,
                dropout=dropout,
                activation='relu',
                batch_first=True  # Use (batch, seq, feature) format
            )
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_layers
            )
            self.decoder_layers = None
        else:
            raise ValueError(f"Invalid positional_encoding: {positional_encoding}. "
                           f"Must be 'rope' or 'sinusoidal'.")
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights using Xavier initialization (consistent with KERMT codebase).
        This is the same initialization used for other KERMT models in nn_utils.py.
        """
        nn.init.xavier_normal_(self.token_embedding.weight)
        nn.init.xavier_normal_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self,
                decoder_input: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the decoder with configurable positional encoding.
        
        :param decoder_input: input token IDs [batch_size, seq_len]
        :param memory: encoder output (molecular latent representation) [batch_size, memory_len, hidden_size]
        :param tgt_mask: causal mask for decoder self-attention [seq_len, seq_len]
        :param tgt_key_padding_mask: padding mask for decoder input [batch_size, seq_len]
        :param memory_key_padding_mask: padding mask for memory [batch_size, memory_len]
        :return: logits for each token [batch_size, seq_len, vocab_size]
        """
        # Embed tokens: [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        tgt_emb = self.token_embedding(decoder_input) * math.sqrt(self.hidden_size)
        
        if self.positional_encoding == 'rope':
            # RoPE: Apply input dropout (positional encoding is applied within attention)
            tgt_emb = self.input_dropout(tgt_emb)  # [batch_size, seq_len, hidden_size]
            
            # Pass through RoPE decoder layers
            # Note: Custom RoPE layers expect FloatTensor masks with additive masking
            # (-inf = masked, 0 = allowed)
            decoder_output = tgt_emb
            for layer in self.decoder_layers:
                decoder_output = layer(
                    tgt=decoder_output,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )  # [batch_size, seq_len, hidden_size]
                
        elif self.positional_encoding == 'sinusoidal':
            # Sinusoidal: Add positional encoding
            tgt_emb = self.pos_encoder(tgt_emb)  # [batch_size, seq_len, hidden_size]
            
            # Pass through standard transformer decoder
            # Note: PyTorch's nn.TransformerDecoder accepts BoolTensor masks
            # (True = masked, False = allowed)
            decoder_output = self.transformer_decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )  # [batch_size, seq_len, hidden_size]
        
        # Project to vocabulary: [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, vocab_size]
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def compute_reconstruction_loss(self,
                                   logits: torch.Tensor,
                                   targets: torch.Tensor,
                                   padding_mask: torch.Tensor = None,
                                   compute_accuracy: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction loss log p(x|z) and optionally token-wise accuracy.
        
        :param logits: predicted logits [batch_size, seq_len, vocab_size]
        :param targets: target token IDs [batch_size, seq_len]
        :param padding_mask: padding mask [batch_size, seq_len], True for padding positions
        :param compute_accuracy: whether to compute token-wise accuracy (for TensorBoard logging)
        :return: tuple of (loss_per_sample, total_loss, accuracy_per_sample, total_accuracy)
                 - loss_per_sample: [batch_size] negative log likelihood per sample
                 - total_loss: scalar, mean loss over batch
                 - accuracy_per_sample: [batch_size] token-wise accuracy per sample (or None if not computed)
                 - total_accuracy: scalar, mean accuracy over batch (or None if not computed)
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for cross entropy: [batch_size * seq_len, vocab_size]
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Compute cross entropy loss (reduction='none' to get per-token loss)
        ce_loss = nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.pad_token_id,
            reduction='none'
        )  # [batch_size * seq_len]
        
        # Reshape back to [batch_size, seq_len]
        ce_loss = ce_loss.reshape(batch_size, seq_len)
        
        # Compute token-wise accuracy only if requested (for TensorBoard logging)
        if compute_accuracy:
            # Get predictions: argmax over vocabulary dimension
            preds = logits.argmax(dim=-1)  # [batch_size, seq_len]
            
            # Compare predictions with targets (element-wise)
            correct = (preds == targets).float()  # [batch_size, seq_len]
        
        # Apply padding mask if provided
        if padding_mask is not None:
            # Mask should be True for valid positions, False for padding
            # But padding_mask from collator is True for padding, so invert it
            valid_mask = ~padding_mask
            ce_loss = ce_loss * valid_mask.float()
            
            # Compute loss per sample (sum over sequence, normalized by sequence length)
            num_valid_tokens = valid_mask.float().sum(dim=1)  # [batch_size]
            loss_per_sample = ce_loss.sum(dim=1) / num_valid_tokens.clamp(min=1.0)  # [batch_size]
            
            # Compute accuracy per sample if requested
            if compute_accuracy:
                correct = correct * valid_mask.float()
                accuracy_per_sample = correct.sum(dim=1) / num_valid_tokens.clamp(min=1.0)  # [batch_size]
            else:
                accuracy_per_sample = None
        else:
            # No mask, average over sequence length
            loss_per_sample = ce_loss.mean(dim=1)  # [batch_size]
            accuracy_per_sample = correct.mean(dim=1) if compute_accuracy else None  # [batch_size]
        
        # Total loss and accuracy: mean over batch
        total_loss = loss_per_sample.mean()
        total_accuracy = accuracy_per_sample.mean() if compute_accuracy else None
        
        # Return negative log likelihood per sample, total loss, and accuracy metrics
        # For CMIM, we want -log p(x|z), which is the cross entropy loss
        return loss_per_sample, total_loss, accuracy_per_sample, total_accuracy


class VocabPredictionModule(nn.Module):
    """
    Modular vocabulary prediction heads for GROVER pretraining.
    Encapsulates atom vocab, bond vocab, and functional group prediction tasks.
    Can be used standalone (in KermtTask) or composed with other modules (in KermtHybridTask).
    """
    
    def __init__(self, args, atom_vocab_size: int, bond_vocab_size: int, fg_size: int):
        """
        Initialize vocabulary prediction module.
        
        :param args: model arguments
        :param atom_vocab_size: size of atom vocabulary
        :param bond_vocab_size: size of bond vocabulary
        :param fg_size: size of functional group labels
        """
        super(VocabPredictionModule, self).__init__()
        
        self.av_task_atom = AtomVocabPrediction(args, atom_vocab_size)
        self.av_task_bond = AtomVocabPrediction(args, atom_vocab_size)
        self.bv_task_atom = BondVocabPrediction(args, bond_vocab_size)
        self.bv_task_bond = BondVocabPrediction(args, bond_vocab_size)
        self.fg_task_all = FunctionalGroupPrediction(args, fg_size)
        
        self.embedding_output_type = args.embedding_output_type
    
    def forward(self, embeddings: Dict, a_scope: List, b_scope: List) -> Dict:
        """
        Compute vocab predictions from embeddings.
        
        :param embeddings: dict with atom_from_atom, atom_from_bond, bond_from_atom, bond_from_bond
        :param a_scope: atom scope for functional group prediction
        :param b_scope: bond scope for functional group prediction
        :return: dict with av_task, bv_task, fg_task predictions
        """
        av_task_pred_atom = self.av_task_atom(embeddings["atom_from_atom"])
        av_task_pred_bond = self.av_task_bond(embeddings["atom_from_bond"])
        bv_task_pred_atom = self.bv_task_atom(embeddings["bond_from_atom"])
        bv_task_pred_bond = self.bv_task_bond(embeddings["bond_from_bond"])
        fg_task_pred_all = self.fg_task_all(embeddings, a_scope, b_scope)
        
        return {
            "av_task": (av_task_pred_atom, av_task_pred_bond),
            "bv_task": (bv_task_pred_atom, bv_task_pred_bond),
            "fg_task": fg_task_pred_all
        }
    
    @staticmethod
    def compute_loss(preds: Dict, targets: Dict, dist_coff: float = 0.1) -> Tuple:
        """
        Compute vocabulary prediction losses.
        
        :param preds: predictions from forward()
        :param targets: dict with av_task, bv_task, fg_task targets
        :param dist_coff: disagreement coefficient
        :return: tuple of (overall_loss, av_loss, bv_loss, fg_loss, av_dist_loss, bv_dist_loss, fg_dist_loss)
        """
        av_task_loss = nn.NLLLoss(ignore_index=0, reduction="mean")
        fg_task_loss = nn.BCEWithLogitsLoss(reduction="mean")
        av_task_dist_loss = nn.MSELoss(reduction="mean")
        fg_task_dist_loss = nn.MSELoss(reduction="mean")
        sigmoid = nn.Sigmoid()

        av_atom_loss, av_bond_loss, av_dist_loss = 0.0, 0.0, 0.0
        fg_atom_from_atom_loss, fg_atom_from_bond_loss, fg_atom_dist_loss = 0.0, 0.0, 0.0
        bv_atom_loss, bv_bond_loss, bv_dist_loss = 0.0, 0.0, 0.0
        fg_bond_from_atom_loss, fg_bond_from_bond_loss, fg_bond_dist_loss = 0.0, 0.0, 0.0

        if preds["av_task"][0] is not None:
            av_atom_loss = av_task_loss(preds['av_task'][0], targets["av_task"])
            fg_atom_from_atom_loss = fg_task_loss(preds["fg_task"]["atom_from_atom"], targets["fg_task"])

        if preds["av_task"][1] is not None:
            av_bond_loss = av_task_loss(preds['av_task'][1], targets["av_task"])
            fg_atom_from_bond_loss = fg_task_loss(preds["fg_task"]["atom_from_bond"], targets["fg_task"])

        if preds["bv_task"][0] is not None:
            bv_atom_loss = av_task_loss(preds['bv_task'][0], targets["bv_task"])
            fg_bond_from_atom_loss = fg_task_loss(preds["fg_task"]["bond_from_atom"], targets["fg_task"])

        if preds["bv_task"][1] is not None:
            bv_bond_loss = av_task_loss(preds['bv_task'][1], targets["bv_task"])
            fg_bond_from_bond_loss = fg_task_loss(preds["fg_task"]["bond_from_bond"], targets["fg_task"])

        if preds["av_task"][0] is not None and preds["av_task"][1] is not None:
            av_dist_loss = av_task_dist_loss(preds['av_task'][0], preds['av_task'][1])
            fg_atom_dist_loss = fg_task_dist_loss(sigmoid(preds["fg_task"]["atom_from_atom"]),
                                                  sigmoid(preds["fg_task"]["atom_from_bond"]))

        if preds["bv_task"][0] is not None and preds["bv_task"][1] is not None:
            bv_dist_loss = av_task_dist_loss(preds['bv_task'][0], preds['bv_task'][1])
            fg_bond_dist_loss = fg_task_dist_loss(sigmoid(preds["fg_task"]["bond_from_atom"]),
                                                  sigmoid(preds["fg_task"]["bond_from_bond"]))

        av_loss = av_atom_loss + av_bond_loss
        bv_loss = bv_atom_loss + bv_bond_loss
        fg_atom_loss = fg_atom_from_atom_loss + fg_atom_from_bond_loss
        fg_bond_loss = fg_bond_from_atom_loss + fg_bond_from_bond_loss

        fg_loss = fg_atom_loss + fg_bond_loss
        fg_dist_loss = fg_atom_dist_loss + fg_bond_dist_loss

        overall_loss = av_loss + bv_loss + fg_loss + dist_coff * av_dist_loss + \
                       dist_coff * bv_dist_loss + fg_dist_loss

        return overall_loss, av_loss, bv_loss, fg_loss, av_dist_loss, bv_dist_loss, fg_dist_loss


class KermtTask(nn.Module):
    """
    The pretrain module for vocabulary-based pretraining (original GROVER).
    """
    def __init__(self, args, kermt, atom_vocab_size, bond_vocab_size, fg_size):
        super(KermtTask, self).__init__()
        self.kermt = kermt
        self.vocab_module = VocabPredictionModule(args, atom_vocab_size, bond_vocab_size, fg_size)
        self.embedding_output_type = args.embedding_output_type

    @staticmethod
    def get_loss_func(args: Namespace) -> Callable:
        """
        The loss function generator.
        :param args: the arguments.
        :return: the loss function for KermtTask.
        """
        def loss_func(preds, targets, dist_coff=args.dist_coff):
            """
            The loss function for KermtTask.
            Delegates to VocabPredictionModule.compute_loss().
            """
            return VocabPredictionModule.compute_loss(preds, targets, dist_coff)

        return loss_func

    def forward(self, graph_batch: List):
        """
        The forward function.
        :param graph_batch: The batched graph input containing tensor components
        :return: Dictionary containing predictions for av_task, bv_task, and fg_task
        """
        _, _, _, _, _, a_scope, b_scope, _ = graph_batch
        a_scope = a_scope.data.cpu().numpy().tolist()

        nvtx.range_push("embedding")
        embeddings = self.kermt(graph_batch)
        nvtx.range_pop()  # embedding

        nvtx.range_push("vocab_prediction")
        preds = self.vocab_module(embeddings, a_scope, b_scope)
        nvtx.range_pop()  # vocab_prediction

        return preds


class KermtCMIMTask(nn.Module):
    """
    The CMIM pretraining module.
    This module uses contrastive learning on molecular embeddings without 
    vocabulary prediction tasks.
    """
    
    def __init__(self, args: Namespace, kermt: KERMTEmbedding = None,
                 latent_dim: int = None, contrastive_temperature: float = 0.1,
                 smiles_vocab_size: int = None):
        """
        Initialize the KermtCMIMTask.
        
        :param args: the arguments containing model configuration
        :param kermt: optional pre-initialized KERMTEmbedding instance
        :param latent_dim: dimension of the latent space. If None, defaults to args.hidden_size
        :param contrastive_temperature: temperature parameter for contrastive loss
        :param smiles_vocab_size: size of SMILES vocabulary (required for decoder)
        """
        super(KermtCMIMTask, self).__init__()
        
        # Initialize latent distribution module
        self.latent_dist = KERMTLatentDistribution(args, kermt=kermt, latent_dim=latent_dim)
        self.contrastive_temperature = contrastive_temperature
        self.embedding_output_type = args.embedding_output_type
        
        # Initialize decoder for SMILES reconstruction
        if smiles_vocab_size is None:
            raise ValueError("smiles_vocab_size is required for KermtCMIMTask")
        
        # Decoder hidden_size MUST equal latent_dim for cross-attention
        # (memory has shape [batch, 1, latent_dim], decoder expects [batch, 1, hidden_size])
        decoder_hidden_size = self.latent_dist.latent_dim
        
        self.decoder = SMILESTransformerDecoder(
            vocab_size=smiles_vocab_size,
            hidden_size=decoder_hidden_size,
            num_layers=args.decoder_num_layers,
            num_attention_heads=args.decoder_num_attention_heads,
            ffn_hidden_size=args.decoder_ffn_hidden_size,
            dropout=args.decoder_dropout,
            max_seq_len=args.decoder_max_seq_len,
            pad_token_id=0,  # Assuming <pad> is at index 0
            positional_encoding=args.decoder_positional_encoding,
            gate_self_attn=args.decoder_gate_self_attn,
            gate_cross_attn=args.decoder_gate_cross_attn
        )
    
    def get_loss_func(self, args: Namespace) -> Callable:
        """
        The loss function generator for CMIM task with reconstruction.
        
        :param args: the arguments
        :return: the loss function for KermtCMIMTask
        """
        # Get parameters from args
        recon_loss_weight = args.reconstruction_loss_weight
        normalize_gradient = args.normalize_gradient
        normalize_loss = args.normalize_loss
        compute_accuracy = args.tensorboard or bool(getattr(args, 'wandb_project', None))

        def loss_func(preds: Dict[str, torch.Tensor], targets=None):
            """
            The loss function for KermtCMIMTask with reconstruction.
            
            :param preds: dictionary containing latent parameters and decoder outputs
            :param targets: not used (kept for API compatibility)
            :return: tuple of (overall_loss, recon_loss_mean, cmim_loss_mean, 
                              log_p_k1_given_zx_mean, log_q_z_given_x_mean, log_P_z_mean, recon_accuracy_mean)
                     Note: recon_accuracy_mean will be None if args.tensorboard is False
            """
            # Extract latent distribution parameters from forward output
            mean = preds["mean"]  # [batch_size, latent_dim]
            log_scale = preds["log_scale"]  # [batch_size, latent_dim]
            z_latent = preds["z_latent"]  # [batch_size, latent_dim]
            
            # Extract decoder outputs
            decoder_logits = preds["decoder_logits"]  # [batch_size, seq_len, vocab_size]
            decoder_target = preds["decoder_target"]  # [batch_size, seq_len]
            decoder_padding_mask = preds["decoder_padding_mask"]  # [batch_size, seq_len]
            
            # Compute CMIM loss (contrastive + KL divergence) with optional gradient normalization
            loss_dict = compute_cmim_loss(
                mean, log_scale, z_latent,
                contrastive_temperature=self.contrastive_temperature,
                normalize_gradient=normalize_gradient,
                normalize_loss=normalize_loss
            )
            
            # Extract CMIM loss components
            cmim_loss = loss_dict["cmim_loss"]  # [batch_size]
            log_p_k1_given_zx = loss_dict["log_p_k1_given_zx"]  # [batch_size]
            log_q_z_given_x = loss_dict["log_q_z_given_x"]  # [batch_size]
            log_P_z = loss_dict["log_P_z"]  # [batch_size]
            
            # Compute reconstruction loss and optionally accuracy: -log p(x|z)
            # Accuracy is only computed if TensorBoard logging is enabled
            _, recon_loss_mean, _, recon_accuracy_mean = self.decoder.compute_reconstruction_loss(
                logits=decoder_logits,
                targets=decoder_target,
                padding_mask=decoder_padding_mask,
                compute_accuracy=compute_accuracy
            )
            
            # Combine losses: Total = CMIM + weight * Reconstruction
            # CMIM loss is already per-sample, reconstruction loss is also per-sample
            overall_loss = cmim_loss.mean() + recon_loss_weight * recon_loss_mean
            
            # Compute mean for logging
            cmim_loss_mean = cmim_loss.mean()
            log_p_k1_given_zx_mean = log_p_k1_given_zx.mean()
            log_q_z_given_x_mean = log_q_z_given_x.mean()
            log_P_z_mean = log_P_z.mean()
            
            # Return: (total_loss, recon_loss, cmim_loss, log_p_k1, log_q_z, log_P_z, recon_accuracy)
            # Note: recon_accuracy_mean will be None if compute_accuracy=False
            return (overall_loss, recon_loss_mean, cmim_loss_mean,
                   log_p_k1_given_zx_mean, log_q_z_given_x_mean, log_P_z_mean, recon_accuracy_mean)
        
        return loss_func
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass for CMIM pretraining with decoder reconstruction.
        
        :param batch: Dictionary containing batch data from collator with:
                      - 'graph_input': The batched graph for encoder
                      - 'decoder_input': Token IDs for decoder input [batch_size, seq_len]
                      - 'decoder_target': Token IDs for reconstruction targets [batch_size, seq_len]
                      - 'decoder_padding_mask': Padding mask [batch_size, seq_len]
                      - 'causal_mask': Causal attention mask [seq_len, seq_len]
        :return: Dictionary containing latent parameters and decoder outputs
        """
        # Extract graph input for encoder
        graph_batch = batch["graph_input"]
        
        # Sample from latent distribution and get parameters
        z_latent, mean, log_scale = self.latent_dist.sample(graph_batch, return_params=True)
        
        # Prepare memory for decoder: z_latent is [batch_size, latent_dim]
        # Decoder expects memory as [batch_size, memory_len, decoder_hidden_size]
        # where decoder_hidden_size == latent_dim (enforced in __init__)
        # We use molecule-level latent embedding as a single memory token
        memory = z_latent.unsqueeze(1)  # [batch_size, 1, latent_dim]
        
        # Decoder forward pass
        decoder_input = batch["decoder_input"]  # [batch_size, seq_len]
        decoder_padding_mask = batch["decoder_padding_mask"]  # [batch_size, seq_len]
        causal_mask = batch["causal_mask"]  # [seq_len, seq_len]
        
        # Get decoder logits
        decoder_logits = self.decoder(
            decoder_input=decoder_input,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_padding_mask,
            memory_key_padding_mask=None  # No padding in memory (single token)
        )  # [batch_size, seq_len, vocab_size]
        
        # Return latent parameters, decoder outputs, and targets for loss computation
        return {
            "mean": mean,
            "log_scale": log_scale,
            "z_latent": z_latent,
            "decoder_logits": decoder_logits,
            "decoder_target": batch["decoder_target"],
            "decoder_padding_mask": decoder_padding_mask
        }


class KermtHybridTask(nn.Module):
    """
    Hybrid pretraining module combining CMIM and vocabulary prediction objectives.
    
    This module uses:
    - CMIM: Contrastive learning + SMILES reconstruction for molecular representation
    - Vocab: Atom/bond vocabulary prediction (original GROVER objectives)
    
    Both objectives share the same KERMT encoder.
    """
    
    def __init__(self, args: Namespace, kermt: KERMTEmbedding = None,
                 latent_dim: int = None, contrastive_temperature: float = 0.1,
                 smiles_vocab_size: int = None,
                 atom_vocab_size: int = None, bond_vocab_size: int = None, fg_size: int = 85):
        """
        Initialize the KermtHybridTask.
        
        :param args: the arguments containing model configuration
        :param kermt: optional pre-initialized KERMTEmbedding instance
        :param latent_dim: dimension of the latent space. If None, defaults to args.latent_dim
        :param contrastive_temperature: temperature parameter for contrastive loss
        :param smiles_vocab_size: size of SMILES vocabulary (required for decoder)
        :param atom_vocab_size: size of atom vocabulary
        :param bond_vocab_size: size of bond vocabulary
        :param fg_size: size of functional group labels (default 85)
        """
        super(KermtHybridTask, self).__init__()
        
        # Store args for loss function
        self.args = args
        
        # Shared encoder
        self.kermt = kermt if kermt is not None else KERMTEmbedding(args)
        self.embedding_output_type = args.embedding_output_type
        
        # CMIM components
        self.latent_dist = KERMTLatentDistribution(args, kermt=self.kermt, latent_dim=latent_dim)
        self.contrastive_temperature = contrastive_temperature
        
        # Validate vocab sizes
        if smiles_vocab_size is None:
            raise ValueError("smiles_vocab_size is required for KermtHybridTask")
        if atom_vocab_size is None or bond_vocab_size is None:
            raise ValueError("atom_vocab_size and bond_vocab_size are required for KermtHybridTask")
        
        # SMILES decoder for reconstruction
        decoder_hidden_size = self.latent_dist.latent_dim
        self.decoder = SMILESTransformerDecoder(
            vocab_size=smiles_vocab_size,
            hidden_size=decoder_hidden_size,
            num_layers=args.decoder_num_layers,
            num_attention_heads=args.decoder_num_attention_heads,
            ffn_hidden_size=args.decoder_ffn_hidden_size,
            dropout=args.decoder_dropout,
            max_seq_len=args.decoder_max_seq_len,
            pad_token_id=0,
            positional_encoding=args.decoder_positional_encoding,
            gate_self_attn=args.decoder_gate_self_attn,
            gate_cross_attn=args.decoder_gate_cross_attn
        )
        
        # Vocabulary prediction module
        self.vocab_module = VocabPredictionModule(args, atom_vocab_size, bond_vocab_size, fg_size)
        
        # Get loss weight from args (default 1.0 means equal weight)
        self.vocab_loss_weight = getattr(args, 'vocab_loss_weight', 1.0)
    
    def get_loss_func(self, args: Namespace) -> Callable:
        """
        The loss function generator for hybrid CMIM + vocab task.
        
        :param args: the arguments
        :return: the loss function for KermtHybridTask
        """
        recon_loss_weight = args.reconstruction_loss_weight
        normalize_gradient = args.normalize_gradient
        normalize_loss = args.normalize_loss
        compute_accuracy = args.tensorboard or bool(getattr(args, 'wandb_project', None))
        dist_coff = args.dist_coff
        vocab_loss_weight = self.vocab_loss_weight
        
        def loss_func(preds: Dict[str, torch.Tensor], targets: Dict = None):
            """
            The loss function for KermtHybridTask.
            
            :param preds: dictionary containing CMIM outputs and vocab predictions
            :param targets: dictionary containing vocab targets (av_task, bv_task, fg_task)
            :return: tuple of loss values for logging
            """
            # === CMIM Loss ===
            mean = preds["mean"]
            log_scale = preds["log_scale"]
            z_latent = preds["z_latent"]
            decoder_logits = preds["decoder_logits"]
            decoder_target = preds["decoder_target"]
            decoder_padding_mask = preds["decoder_padding_mask"]
            
            # Compute CMIM loss using module-level helper
            loss_dict = compute_cmim_loss(
                mean, log_scale, z_latent,
                contrastive_temperature=self.contrastive_temperature,
                normalize_gradient=normalize_gradient,
                normalize_loss=normalize_loss
            )
            
            cmim_loss = loss_dict["cmim_loss"]
            log_p_k1_given_zx = loss_dict["log_p_k1_given_zx"]
            log_q_z_given_x = loss_dict["log_q_z_given_x"]
            log_P_z = loss_dict["log_P_z"]
            
            # Compute reconstruction loss
            _, recon_loss_mean, _, recon_accuracy_mean = self.decoder.compute_reconstruction_loss(
                logits=decoder_logits,
                targets=decoder_target,
                padding_mask=decoder_padding_mask,
                compute_accuracy=compute_accuracy
            )
            
            # CMIM total (contrastive + KL + reconstruction)
            cmim_total = cmim_loss.mean() + recon_loss_weight * recon_loss_mean
            
            # === Vocab Loss ===
            vocab_preds = preds["vocab_preds"]
            vocab_overall, av_loss, bv_loss, fg_loss, av_dist_loss, bv_dist_loss, fg_dist_loss = \
                VocabPredictionModule.compute_loss(vocab_preds, targets, dist_coff)
            
            # === Combined Loss ===
            overall_loss = cmim_total + vocab_loss_weight * vocab_overall
            
            # Return all components for logging
            # Order: (overall, cmim_total, recon, cmim_only, log_p_k1, log_q_z, log_P_z, recon_acc,
            #         vocab_total, av, bv, fg, av_dist, bv_dist, fg_dist)
            return (
                overall_loss,
                cmim_total,
                recon_loss_mean,
                cmim_loss.mean(),
                log_p_k1_given_zx.mean(),
                log_q_z_given_x.mean(),
                log_P_z.mean(),
                recon_accuracy_mean,
                vocab_overall,
                av_loss,
                bv_loss,
                fg_loss,
                av_dist_loss,
                bv_dist_loss,
                fg_dist_loss
            )
        
        return loss_func
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass for hybrid CMIM + vocab pretraining.
        
        :param batch: Dictionary containing:
                      - 'graph_input': The batched graph for encoder
                      - 'decoder_input': Token IDs for decoder input
                      - 'decoder_target': Token IDs for reconstruction targets
                      - 'decoder_padding_mask': Padding mask
                      - 'causal_mask': Causal attention mask
        :return: Dictionary containing CMIM outputs and vocab predictions
        """
        # Extract graph input
        graph_batch = batch["graph_input"]
        _, _, _, _, _, a_scope, b_scope, _ = graph_batch
        a_scope_list = a_scope.data.cpu().numpy().tolist()
        
        # Get embeddings from shared encoder
        embeddings = self.kermt(graph_batch)
        
        # === CMIM path ===
        # Sample from latent distribution
        z_latent, mean, log_scale = self.latent_dist.sample(graph_batch, return_params=True)
        
        # Decoder forward pass
        memory = z_latent.unsqueeze(1)
        decoder_input = batch["decoder_input"]
        decoder_padding_mask = batch["decoder_padding_mask"]
        causal_mask = batch["causal_mask"]
        
        decoder_logits = self.decoder(
            decoder_input=decoder_input,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_padding_mask,
            memory_key_padding_mask=None
        )
        
        # === Vocab path ===
        vocab_preds = self.vocab_module(embeddings, a_scope_list, b_scope)
        
        # Return combined output
        return {
            # CMIM outputs
            "mean": mean,
            "log_scale": log_scale,
            "z_latent": z_latent,
            "decoder_logits": decoder_logits,
            "decoder_target": batch["decoder_target"],
            "decoder_padding_mask": decoder_padding_mask,
            # Vocab outputs
            "vocab_preds": vocab_preds
        }


class KermtFpGeneration(nn.Module):
    """
    KermtFpGeneration class.
    It loads the pre-trained model and produce the fingerprints for input molecules.
    """
    def __init__(self, args):
        """
        Init function.
        :param args: the arguments.
        """
        super(KermtFpGeneration, self).__init__()

        self.fingerprint_source = args.fingerprint_source
        self.iscuda = args.cuda

        self.kermt = KERMTEmbedding(args)
        self.readout = Readout(rtype="mean", hidden_size=args.hidden_size)

    def forward(self, batch, features_batch):
        """
        The forward function.
        It takes graph batch and molecular feature batch as input and produce the fingerprints of this molecules.
        :param batch:
        :param features_batch:
        :return:
        """
        _, _, _, _, _, a_scope, b_scope, _ = batch

        output = self.kermt(batch)
        # Share readout
        mol_atom_from_bond_output = self.readout(output["atom_from_bond"], a_scope)
        mol_atom_from_atom_output = self.readout(output["atom_from_atom"], a_scope)

        if self.fingerprint_source == "bond" or self.fingerprint_source == "both":
            mol_bond_from_atom_output = self.readout(output["bond_from_atom"], b_scope)
            mol_bond_from_bodd_output = self.readout(output["bond_from_bond"], b_scope)

        if features_batch[0] is not None:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if self.iscuda:
                features_batch = features_batch.cuda()
            features_batch = features_batch.to(output["atom_from_atom"])
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
        else:
            features_batch = None

        if self.fingerprint_source == "atom":
            fp = torch.cat([mol_atom_from_atom_output, mol_atom_from_bond_output], 1)
        elif self.fingerprint_source == "bond":
            fp = torch.cat([mol_bond_from_atom_output, mol_bond_from_bodd_output], 1)
        else:
            # the both case.
            fp = torch.cat([mol_atom_from_atom_output, mol_atom_from_bond_output,
                            mol_bond_from_atom_output, mol_bond_from_bodd_output], 1)
        if features_batch is not None:
            fp = torch.cat([fp, features_batch], 1)
        return fp


class KermtFinetuneTask(nn.Module):
    """
    The finetune
    """
    def __init__(self, args):
        super(KermtFinetuneTask, self).__init__()

        self.hidden_size = args.hidden_size
        self.iscuda = args.cuda

        self.kermt = KERMTEmbedding(args)

        if args.self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=self.hidden_size,
                                   attn_hidden=args.attn_hidden,
                                   attn_out=args.attn_out)
        else:
            self.readout = Readout(rtype="mean", hidden_size=self.hidden_size)

        if args.ffn_num_task_specific_layers > 0:
            ffn_output_size = args.ffn_task_specific_hidden_size
        else:
            ffn_output_size = args.output_size
        self.mol_atom_from_atom_ffn = self.create_ffn(args, ffn_output_size)
        self.mol_atom_from_bond_ffn = self.create_ffn(args, ffn_output_size)

        if args.ffn_num_task_specific_layers > 0:
            self.mol_atom_from_atom_ffn_task_specific = nn.ModuleList([self.create_task_specific_ffn(ffn_output_size, args.ffn_num_task_specific_layers,  args.ffn_task_specific_hidden_size, args.dropout, args.activation) for _ in range(args.num_tasks)])

            self.mol_atom_from_bond_ffn_task_specific = nn.ModuleList([self.create_task_specific_ffn(ffn_output_size, args.ffn_num_task_specific_layers,  args.ffn_task_specific_hidden_size, args.dropout, args.activation) for _ in range(args.num_tasks)])
        else:
            self.mol_atom_from_atom_ffn_task_specific = None
            self.mol_atom_from_bond_ffn_task_specific = None
    


        #self.ffn = nn.ModuleList()
        #self.ffn.append(self.mol_atom_from_atom_ffn)
        #self.ffn.append(self.mol_atom_from_bond_ffn)

        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()


    def create_task_specific_ffn(self, input_size: int, num_layers: int, hidden_size: int, dropout, activation):
        dropout_layer = nn.Dropout(dropout)
        activation_layer = get_activation_function(activation)
        if num_layers == 1:
            ffn_ts = [dropout_layer, nn.Linear(input_size, 1)]
        else:
            ffn_ts = []
            for _ in range(num_layers - 1):
                ffn_ts.extend([
                    activation_layer,
                    dropout_layer,
                    nn.Linear(hidden_size, hidden_size)
                ])
            ffn_ts.extend([
                activation_layer,
                dropout_layer,
                nn.Linear(hidden_size, 1)
            ])

        return nn.Sequential(*ffn_ts)


    def create_ffn(self, args: Namespace, output_size: int):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        # Default ffn_hidden_size to hidden_size if not specified
        ffn_hidden_size = args.ffn_hidden_size if args.ffn_hidden_size is not None else args.hidden_size
        
        # Note: args.features_dim is set according the real loaded features data
        if args.features_only:
            first_linear_dim = args.features_size + args.features_dim
        else:
            if args.self_attention:
                first_linear_dim = args.hidden_size * args.attn_out
                # TODO: Ad-hoc!
                # if args.use_input_features:
                # first_linear_dim += args.features_dim
                # TODO(sveccham): Verify that this is correct.
                first_linear_dim += args.features_size 
            else:
                # TODO(sveccham): Verify that this is correct.
                first_linear_dim = args.hidden_size + args.features_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(ffn_hidden_size, ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(ffn_hidden_size, output_size),
            ])

        # Create FFN model
        return nn.Sequential(*ffn)

    @staticmethod
    def get_loss_func(args):
        def loss_func(preds, targets,
                      dt=args.dataset_type,
                      dist_coff=args.dist_coff):

            if dt == 'classification':
                pred_loss = nn.BCEWithLogitsLoss(reduction='none')
            elif dt == 'regression':
                pred_loss = nn.MSELoss(reduction='none')
            else:
                raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

            # print(type(preds))
            # TODO: Here, should we need to involve the model status? Using len(preds) is just a hack.
            if type(preds) is not tuple:
                # in eval mode.
                return pred_loss(preds, targets)

            # in train mode.
            dist_loss = nn.MSELoss(reduction='none')
            # dist_loss = nn.CosineSimilarity(dim=0)
            # print(pred_loss)

            dist = dist_loss(preds[0], preds[1])
            pred_loss1 = pred_loss(preds[0], targets)
            pred_loss2 = pred_loss(preds[1], targets)
            return pred_loss1 + pred_loss2 + dist_coff * dist

        return loss_func

    def forward(self, batch, features_batch):
        _, _, _, _, _, a_scope, _, _ = batch

        output = self.kermt(batch)
        # Share readout
        mol_atom_from_bond_output = self.readout(output["atom_from_bond"], a_scope)
        mol_atom_from_atom_output = self.readout(output["atom_from_atom"], a_scope)

        if features_batch[0] is not None:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if self.iscuda:
                features_batch = features_batch.cuda()
            features_batch = features_batch.to(output["atom_from_atom"])
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
        else:
            features_batch = None


        if features_batch is not None:
            mol_atom_from_atom_output = torch.cat([mol_atom_from_atom_output, features_batch], 1)
            mol_atom_from_bond_output = torch.cat([mol_atom_from_bond_output, features_batch], 1)

        if self.training:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            atom_preds = []
            bond_preds = []
            if self.mol_atom_from_atom_ffn_task_specific is not None: # TODO
                for _, task_layer in enumerate(self.mol_atom_from_atom_ffn_task_specific):
                    atom_preds.append(task_layer(atom_ffn_output))
                for _, task_layer in enumerate(self.mol_atom_from_bond_ffn_task_specific):
                    bond_preds.append(task_layer(bond_ffn_output))
                return torch.hstack(atom_preds), torch.hstack(bond_preds)
            else:
                return atom_ffn_output, bond_ffn_output
        else:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)

            if self.mol_atom_from_atom_ffn_task_specific is not None:
                atom_preds_list = []
                bond_preds_list = []
                for _, task_layer in enumerate(self.mol_atom_from_atom_ffn_task_specific):
                    atom_preds_list.append(task_layer(atom_ffn_output))
                for _, task_layer in enumerate(self.mol_atom_from_bond_ffn_task_specific):
                    bond_preds_list.append(task_layer(bond_ffn_output))
                atom_preds = torch.hstack(atom_preds_list)
                bond_preds = torch.hstack(bond_preds_list)
                if self.classification:
                    atom_preds = self.sigmoid(atom_preds)
                    bond_preds = self.sigmoid(bond_preds)
                    output = (atom_preds + bond_preds) / 2
                else:
                    output = (atom_preds + bond_preds) / 2
            else:
                if self.classification:
                    atom_ffn_output = self.sigmoid(atom_ffn_output)
                    bond_ffn_output = self.sigmoid(bond_ffn_output)
                    output = (atom_ffn_output + bond_ffn_output) / 2
                else:
                    output = (atom_ffn_output + bond_ffn_output) / 2

        return output
