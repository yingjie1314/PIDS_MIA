import torch.nn as nn
import torch.nn.functional as F

from pidsmaker.utils.utils import compute_class_weights


class EdgeTypePrediction(nn.Module):
    def __init__(self, decoder, loss_fn, balanced_loss, edge_type_dim):
        super(EdgeTypePrediction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.balanced_loss = balanced_loss
        self.edge_type_dim = edge_type_dim

    def forward(self, h_src, h_dst, edge_type, inference, **kwargs):
        h = self.decoder(h_src=h_src, h_dst=h_dst)

        class_weights = (
            compute_class_weights(edge_type, num_classes=self.edge_type_dim)
            if self.balanced_loss
            else None
        )

        edge_type_classes = edge_type.argmax(dim=1)
        loss = self.loss_fn(h, edge_type_classes, inference=inference, class_weights=class_weights)

        out = {"loss": loss}
        if inference:
            # `h` are this head's per-edge class logits over edge types (E, num_edge_types). The
            # detector is already a supervised classifier here, so this is a native posterior --
            # what posterior-based MIA (He'21 et al.) expects. Logits are kept alongside because
            # softmax discards scale, which carries membership signal. Inference-only, so training
            # is bit-for-bit unaffected.
            out["logits"] = h.detach()
            out["posterior"] = F.softmax(h, dim=1).detach()
        return out
