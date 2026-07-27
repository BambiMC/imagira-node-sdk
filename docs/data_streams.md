# The four data streams

Four independent streams flow between nodes. A connection carries exactly one
of them, and each has its own port colour in the editor.

| Stream | Colour | Python type | Shape |
|---|---|---|---|
| image | blue | `np.uint8` | `(H, W, 3)` RGB |
| mask | orange | `np.bool_` | `(H, W)` |
| samples | red | `dict` | see below |
| data | green | `dict` | free-form |

Your node declares which it accepts and produces with the port flags — see
[authoring.md](authoring.md#ports).

## image

`np.uint8`, `(H, W, 3)`, RGB (not BGR).

Incoming images should already be in that form, but upstream nodes are written
by other people. `to_rgb()` normalises anything reasonable:

```python
from imagira_node_sdk import to_rgb

src = to_rgb(image)   # (H,W) / (H,W,1) / (H,W,3) / (H,W,4) / float → uint8 (H,W,3)
```

It returns already-correct arrays unchanged (no copy), broadcasts grayscale,
drops the alpha channel from RGBA, and clips-then-casts float input.

**Return `uint8`.** A float image silently wraps or clips somewhere
downstream — end with `np.clip(out, 0, 255).astype(np.uint8)`.

## mask

`np.bool_`, `(H, W)` — same height and width as the image, no channel axis.

`None` means "no mask", which is the common case. Handle it:

```python
def execute(self, image, mask=None, data=None, context=None):
    ...
    return {"image": out, "mask": mask}      # pass it through if unchanged
```

**Return `np.bool_`, not `uint8` 0/255.** A uint8 mask is truthy everywhere it
isn't 0, which quietly turns a soft mask into a hard one.

To confine your effect to the mask, use `mask_blend()` — it alpha-composites
your result over the original using the mask as the blend weight, accepts bool
or 0..1 float masks, and returns your result untouched when `mask is None`:

```python
from imagira_node_sdk import mask_blend

result = mask_blend(src, result, mask)
```

`mask_blend` raises if the mask doesn't cover your result — which happens when a
node resizes or crops the image and then blends against the *original* mask. If
you change the image's size, return the resized mask or `mask=None`; never pass
the incoming one through unchanged.

### Reading a mask: convert before you index

You must *return* `bool`, but what you *receive* can be `bool`, `uint8` (0/1 or
0/255), or a float soft matte in [0, 1] — mattes come from keyers and feathering
nodes and are the whole point of soft-edge compositing. Only `bool` works as a
selection:

```python
image[mask]          # uint8 mask → INTEGER indexing: wrong pixels, or
                     # "index 255 is out of bounds for axis 0 with size 96"
~mask                # uint8 mask → bitwise complement (254, not False)
                     # float mask → TypeError: ufunc 'invert' not supported
```

Convert first with `as_bool_mask()` whenever you index with the mask or use a
bitwise operator (`~ & |`) on it. Float mattes threshold at 0.5:

```python
from imagira_node_sdk import as_bool_mask

sel = as_bool_mask(mask)          # None stays None
out[sel] = fill_colour
```

Use the raw float via `mask_blend` when you want the soft edge; use
`as_bool_mask` when you need a hard yes/no per pixel. Imagira's own fuzzer found
this class of bug in nine shipped nodes at once — it is the single easiest
mistake to make against the mask stream.

## samples

A `dict` describing connected regions ("islands") of the image, with the pixels
and positions belonging to each. This is what colour-analysis and
region-processing nodes pass between each other.

```python
samples = {
    "islands": [
        {
            "pixels":     np.ndarray,   # (N, 3) uint8 — the island's pixel colours
            "positions":  np.ndarray,   # (N, 2) int   — [row, col] per pixel
            "label":      int,          # stable cross-node join key
            "area":       int,          # N
            "mean_color": [R, G, B],
        },
        ...
    ],
    "image_size": (H, W),
}
```

`label` is the important field: it's stable across nodes, so a node downstream
can join its own per-island results back to the islands it was given.

`imagira_node_sdk.testing.make_samples(image, mask)` builds a minimal
one-island payload in this shape for tests.

To *receive* samples, a node needs both `accepts_samples = True` and a `samples`
parameter on `execute()` — it's an opt-in fifth argument, not part of the default
signature. See [authoring.md](authoring.md#the-last-three-arguments-are-opt-in--this-trips-people-up).

## data

A free-form `dict`. Keys are entirely producer-defined — e.g.
`{"color": [r, g, b], "value": 0.5}`, `{"fps": 30}`, whatever a node wants to
hand downstream.

Because it's untyped by design, the interesting part is that **users can reach
into it from any param field** with a `{{ }}` template, without your node
knowing anything about it:

```
{{ $json.article.title }}.png     # in a filename param
{{ $json.fps }}                   # in a numeric param — stays a real number
```

`$json` and `$data` are aliases for the incoming data payload. A field that is
*exactly* one `{{ … }}` expression returns the raw Python value (so a number
stays a number); a field with the expression embedded in other text stringifies
it in place (dicts and lists as indented JSON).

There's a second namespace, `$nodes`, which reads another node's *configured
param value* by id — no wire required, since a param value is static graph
configuration:

```
{{ $nodes["chroma_key_1"].tolerance }}
```

Values there are the raw stored value: if the referenced node's own param is
itself a template, you get that literal `{{ … }}` text back rather than a
further-resolved value. That's deliberate — resolving it would mean evaluating
another node's expression against the wrong data context, and could cycle.

All of this works as long as your node reads params through
`self.params.get(key, default)`. See
[authoring.md](authoring.md#params).

## Return dict

```python
{"image": ..., "mask": ..., "data": ..., "samples": ..., "_iterations": ...}
```

Those five keys are what the pipeline reads. Anything else is dropped without a
word — a typo'd `"images"` is an easy and invisible bug, which is why
`run_node()` in the test kit rejects unrecognised keys.
