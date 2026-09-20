"""Byte-bounded scoring-only cache; importing it does not import PyTorch."""
from __future__ import annotations

from collections import OrderedDict
import sys


class OutputCache:
    def __init__(self, max_bytes):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("cache size must be a nonnegative integer")
        self.max_bytes = max_bytes
        self.bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.forwards = 0
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return None
        self.hits += 1
        self.data.move_to_end(key)
        return self.data[key][0]

    def put(self, key, value, payload_bytes):
        size = payload_bytes + sum(sys.getsizeof(part) for part in key) + sys.getsizeof(key) + 512
        if size > self.max_bytes:
            return
        if key in self.data:
            self.bytes -= self.data.pop(key)[1]
        while self.data and (self.bytes + size > self.max_bytes or len(self.data) >= 8192):
            self.bytes -= self.data.popitem(last=False)[1][1]
        self.data[key] = (value, size)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)


def wrap_scoring_model(model, max_bytes, torch):
    """Reuse full logits across bounded upstream score() chunks.

    The wrapper is private to peptide PLL scoring: cached outputs intentionally
    contain logits only. All model weights/context/device are fixed for its
    process lifetime. Unknown forward arguments bypass the cache completely.
    """
    cache = OutputCache(max_bytes)

    class CachedScoringModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = model

        def forward(self, tokens, *args, **kwargs):
            known = set(kwargs) <= {"repr_layers", "return_contacts"}
            layers = kwargs.get("repr_layers")
            if (not max_bytes or args or not known or kwargs.get("return_contacts", False)
                    or (layers is not None and (not isinstance(layers, (list, tuple))
                                               or any(type(layer) is not int for layer in layers)))):
                cache.forwards += 1
                return self.backbone(tokens, *args, **kwargs)
            # Include shape/dtype/device and all supported kwargs. A different
            # mask changes the integer token bytes; extra masks bypass above.
            array = tokens.detach().cpu().contiguous().numpy()
            key = (str(tokens.dtype), str(tokens.device), tuple(tokens.shape), array.tobytes(),
                   None if layers is None else tuple(layers), kwargs.get("return_contacts", False))
            cached = cache.get(key)
            if cached is not None:
                return {"logits": cached.to(tokens.device)}
            cache.forwards += 1
            output = self.backbone(tokens, **kwargs)
            logits = output["logits"]
            payload = logits.numel() * logits.element_size()
            if payload <= max_bytes:
                # Own the cached storage, detach autograd, and retain no full
                # hidden representations or GPU allocations between chunks.
                stored = logits.detach().cpu().contiguous().clone()
                cache.put(key, stored, payload)
            return {"logits": logits}

    return CachedScoringModel().eval(), cache
