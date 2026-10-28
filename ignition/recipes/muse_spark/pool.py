"""Paged full-attention histories and bounded sliding-attention continuation rows."""

from dataclasses import dataclass
import math

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

PAGE_TOKENS = 128


def page_specs(config, shard):
    for layer in range(config.layers):
        geometry = config.attention(layer)
        if geometry.window < 0:
            _, heads = geometry.local(shard)
            for kind, dim in (("key", geometry.key_dim), ("value", geometry.value_dim)):
                yield f"layer{layer}.{kind}", (PAGE_TOKENS * heads * dim * 2,)


def page_bytes(config, shard):
    return sum(shape[0] for _, shape in page_specs(config, shard))


def continuation_layout(config, shard):
    regions = []
    for layer in range(config.layers):
        geometry = config.attention(layer)
        if geometry.window > 0:
            _, heads = geometry.local(shard)
            for kind, dim in (("key", geometry.key_dim), ("value", geometry.value_dim)):
                regions.append((f"layer{layer}.{kind}", (geometry.window, heads, dim), torch.bfloat16))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config, shard))
    regions.extend((("last_hidden", (config.text["hidden_size"],), torch.bfloat16),
                    ("consumed_tokens", (1,), torch.int64)))
    layout = StateLayout.from_specs(regions)
    # TMA stages divide the stride between circular histories in adjacent rows.
    alignment = math.lcm(256, *(math.prod(shape) * dtype.itemsize
        for name, shape, dtype in regions if name.startswith("layer") and dtype == torch.bfloat16))
    return StateLayout(layout.regions, (layout.row_bytes + alignment - 1) // alignment * alignment)


@dataclass(frozen=True)
class ContinuationPool(PagedStateStorage):
    config: object
    shard: object

    @classmethod
    def allocate(cls, config, shard, slots, pages, device, *, storage=None, maximum_pages=None, maximum_rows=None,
                 backed_slots=None):
        layout = continuation_layout(config, shard)
        backing = PagedStateStorage.allocate(layout, page_specs(config, shard), slots, pages, device,
            storage=storage, maximum_pages=maximum_pages, maximum_rows=maximum_rows, backed_slots=backed_slots)
        return cls(backing.backing, backing.layout, backing.memory_capacity,
                   backing.pages, backing.reserved_backing, config, shard)

    def cache(self, layer, kind):
        geometry = self.config.attention(layer)
        name = f"layer{layer}.{kind}"
        if geometry.window > 0:
            return self.reserved_region(name)
        _, heads = geometry.local(self.shard)
        dim = geometry.key_dim if kind == "key" else geometry.value_dim
        return self.pages[name].view(torch.bfloat16).view(-1, PAGE_TOKENS, heads, dim)

    def window_page_stride(self, layer):
        geometry = self.config.attention(layer)
        _, heads = geometry.local(self.shard)
        record = PAGE_TOKENS * heads * geometry.key_dim * 2
        if self.layout.row_bytes % record:
            raise ValueError("circular continuation rows must align to physical KV pages")
        return self.layout.row_bytes // record

    def window_pages(self, layer, kind):
        """Expose circular regions as pages with the continuation row stride intact."""
        geometry = self.config.attention(layer)
        _, heads = geometry.local(self.shard)
        dim = geometry.key_dim if kind == "key" else geometry.value_dim
        region = self.layout.region(f"layer{layer}.{kind}")
        row_pages = self.window_page_stride(layer)
        pages = (self.reserved_backing.shape[0] - 1) * row_pages + geometry.window // PAGE_TOKENS
        return self.reserved_backing.view(torch.bfloat16).as_strided(
            (pages, PAGE_TOKENS, heads, dim), (PAGE_TOKENS * heads * dim, heads * dim, dim, 1),
            storage_offset=region.offset // 2)
