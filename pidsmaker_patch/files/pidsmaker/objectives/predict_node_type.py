import torch.nn as nn
import torch.nn.functional as F

from pidsmaker.utils.utils import compute_class_weights


class NodeTypePrediction(nn.Module):
    def __init__(self, decoder, loss_fn, balanced_loss, node_type_dim):
        super(NodeTypePrediction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.balanced_loss = balanced_loss
        self.node_type_dim = node_type_dim

    def forward(self, h, node_type, inference, **kwargs):
        h_hat = self.decoder(h)

        class_weights = (
            compute_class_weights(node_type, num_classes=self.node_type_dim)
            if self.balanced_loss
            else None
        )

        node_type_classes = node_type.argmax(dim=1)
        loss = self.loss_fn(h_hat, node_type_classes, inference=inference, weight=class_weights)

        # NOTE: "out" softmaxes `h` (the encoder embedding), not `h_hat` (this head's class logits).
        # That is only equivalent because every detector using this objective sets `decoder: none`,
        # making h_hat is h. Left untouched -- it feeds detection scoring (inference_loop.py:168,198)
        # and changing it would move detection numbers. The posterior below uses `h_hat`, which is
        # correct for any decoder.
        out = {"loss": loss, "out": F.log_softmax(h, dim=1)}
        if inference:
            # Native per-node posterior over node types (N, num_node_types): this detector's own
            # trained classification head. Logits kept because softmax discards scale.
            out["logits"] = h_hat.detach()
            out["posterior"] = F.softmax(h_hat, dim=1).detach()
        return out
