import torch
import torch.nn as nn


class GMAEFeatReconstruction(nn.Module):
    def __init__(self, decoder, loss_fn, mask_rate, embed_dim):
        super(GMAEFeatReconstruction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.mask_rate = mask_rate
        # Learnable mask token initialized to zeros, matching MAGIC-main
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    def mask_input(self, x):
        """
        Masks the input features BEFORE encoding.
        Returns:
            x_masked: The input features with masked nodes replaced by token
            mask_nodes: Indices of masked nodes
        """
        num_nodes = x.shape[0]
        num_mask_nodes = int(self.mask_rate * num_nodes)
        perm = torch.randperm(num_nodes, device=x.device)
        mask_nodes = perm[:num_mask_nodes]

        x_masked = x.clone()
        x_masked[mask_nodes] = self.mask_token.to(x.device)

        return x_masked, mask_nodes

    def forward(self, x, h, edge_index, inference, **kwargs):
        # Recover mask_nodes passed from Model.forward
        mask_nodes = kwargs.get("mask_nodes", None)

        # During inference or if no mask was applied, we might not have mask_nodes.
        # But for training this objective, we expect them.
        if mask_nodes is None and not inference:
            raise ValueError(
                "GMAEFeatReconstruction objective requires masked nodes, but mask_nodes is None."
            )

        batch = kwargs.get("batch")
        edge_feats = getattr(batch, "edge_feats", None) if batch else None

        recon = self.decoder(h, edge_index, edge_feats=edge_feats)

        # Calculate loss only on masked nodes
        if mask_nodes is not None:
            x_init = x[mask_nodes].to(h.device)
            x_rec = recon[mask_nodes].to(h.device)
        else:
            # If inference, maybe we want global loss? Or just 0 as per previous code
            x_init = x
            x_rec = recon

        if inference:
            # Per-node reconstruction error over ALL nodes. At inference no masking is
            # applied (see Model._embed_nodes: masking is train-only), so mask_nodes is
            # None and x_init/x_rec above are the full unmasked features and their
            # reconstruction. sce_loss(..., inference=True) is NOT reduced, so this is a
            # per-node (N,) score: a memorized/member node reconstructs with lower error.
            #
            # This is the MIA membership signal, the direct analog of NodLink's per-node
            # reconstruction loss. It previously returned zeros because MAGIC's *detection*
            # uses the kNN magic_score (computed downstream in test_node_level, not here),
            # not this reconstruction error -- so nothing consumed it. The MIA dump does.
            return {"loss": self.loss_fn(x_rec, x_init, inference=True)}

        loss = self.loss_fn(x_rec, x_init, inference=inference)

        return {"loss": loss}
