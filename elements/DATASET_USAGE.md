# Elements Dataset Guide

This document explains how the dataset generation code in `elements` works, with a focus on the dataset-related classes in `elements/classes.py`.

## What the package generates

The package creates synthetic RGB images containing multiple geometric objects ("elements").
Each object has three concept attributes:

- shape
- color
- texture

Images are generated with controlled randomness (seeded), and can be labeled according to user-defined class rules.

## Concept vocabularies

### Shapes
Defined in `elements/shapes.py`:

- `square`
- `circle`
- `triangle`
- `cross`
- `plus`

### Colors
Defined in `elements/colors.py`:

- `red`
- `green`
- `blue`
- `yellow`
- `cyan`
- `magenta`

### Textures
Defined in `elements/textures.py` and referenced through naming convention in `Element`:

- `solid` (special case: no texture overlay)
- `spots_random`
- `spots_regular`
- `spots_polka`
- `spots_chequerboard`
- `stripes_horizontal`
- `stripes_vertical`
- `stripes_diagonal`
- `stripes_diagonal_alt`

Texture names are split on `_` and mapped as `<family>_<variant>`, for example:

- `spots_polka` -> family `spots`, variant `polka`
- `stripes_diagonal` -> family `stripes`, variant `diagonal`

## Core generation classes

## `Element`
Represents one object patch (a single shape instance).

Constructor signature:

```python
Element(size, shape, color, texture=None, color_seed=None, texture_seed=None)
```

Parameters:

- `size` (`int`): patch width and height in pixels.
- `shape` (`str`): one of the shape names above.
- `color` (`str`): one of the color names above.
- `texture` (`str | None`): texture name. `"solid"` is treated as no texture.
- `color_seed` (`int | None`): controls color intensity variation.
- `texture_seed` (`int | None`): controls stochastic texture variation.

Behavior notes:

- Shape mask is created first.
- Texture is applied second (if not solid).
- Color is applied third, with a random brightness multiplier from `color_adjustment()`.
- Background pixels are forced to white.

## `ElementImage`
Represents one full image containing multiple `Element` objects placed on a white canvas.

Constructor signature:

```python
ElementImage(
    elements,
    size=224,
    loc_seed=None,
    loc_restrictions=None,
    place_remaining_randomly=True,
)
```

Parameters:

- `elements` (`List[Element]`): objects to place.
- `size` (`int`): canvas size (square image of `size x size`).
- `loc_seed` (`int | None`): random seed for placement.
- `loc_restrictions` (`List[List[str]] | None`): placement constraints, such as `[["<127", ">0"]]`.
- `place_remaining_randomly` (`bool`): behavior when not all objects can be placed without overlap/restriction violations.

Placement behavior:

- Objects are placed without overlap when possible.
- If placement fails after retries:
  - `place_remaining_randomly=True`: remaining objects are still placed randomly, even if overlap may occur.
  - `place_remaining_randomly=False`: remaining objects are marked missing (`None` location), and ignored in class membership checks.

Location restriction format:

- Each restriction is a 2-item list of comparator strings, for example `"<127"`, `">0"`.
- Internal coordinate handling follows the implementation in `check_fail_loc_restriction`; use the provided keywords (`left`, `right`, `top`, `bot`) unless you need custom logic.

## `ElementDataset`
Main dataset class used for model training/inference.
Generates synthetic samples on demand.

Constructor signature:

```python
ElementDataset(
    allowed,
    class_configs,
    n,
    img_size,
    element_n,
    element_size,
    element_size_delta,
    element_seed,
    loc_seed,
    allowed_combinations=None,
    loc_restrictions=None,
    place_remaining_randomly=True,
)
```

Parameters:

- `allowed` (`dict`): allowed concept values.
  - Required keys: `"shapes"`, `"colors"`, `"textures"`.
  - Values are lists of allowed names.
- `class_configs` (`List[dict]`): class definitions used to compute labels.
  - Typical keys: `shape`, `color`, `texture`.
  - Values can be concrete values or `None` as wildcard.
- `n` (`int`): dataset length.
- `img_size` (`int`): final image size.
- `element_n` (`int`): number of objects per image.
- `element_size` (`int`): central object size target.
- `element_size_delta` (`int`): random size jitter around `element_size`.
- `element_seed` (`int`): seed stream for concept sampling.
- `loc_seed` (`int`): seed stream for placement sampling.
- `allowed_combinations` (`List[Tuple[str, str, str]] | List[List[str]] | None`): optional whitelist of valid `(shape, color, texture)` triples.
- `loc_restrictions` (`List[List[str]] | None`): optional global location constraints for all placed objects.
- `place_remaining_randomly` (`bool`): placement fallback behavior.

Returned sample format (`__getitem__`):

```python
[image_tensor, class_one_hot]
```

- `image_tensor`: PyTorch tensor (`C x H x W`) from `ToTensor()`.
- `class_one_hot`: multi-hot vector over `class_configs` (an image can belong to multiple classes).

Important detail for `element_size_delta`:

- Sizes are sampled via `rng.integers(low, high)` with
  - `low = element_size - element_size_delta`
  - `high = element_size + element_size_delta`
- The upper bound is exclusive.

## `GroupedElementDataset`
Subclass of `ElementDataset` that precomputes class membership over all `n` samples and lets you index by class.

Usage pattern:

1. Create dataset.
2. Set `current_label` to target class index.
3. Iterate samples from only that class.

If `current_label` is `None`, it behaves like the full dataset.

## `ConceptElementDatasetCreator`
Utility wrapper that returns datasets specialized by query string.

Constructor:

```python
ConceptElementDatasetCreator(
    allowed,
    class_configs,
    dataset_kwargs,
    allowed_combinations=None,
)
```

`dataset_kwargs` typically contains:

- `n`
- `img_size`
- `element_n`
- `element_size`
- `element_size_delta`
- `element_seed`
- `loc_seed`

`__call__(concept)` supports:

- Class index string, e.g. `"13"`: returns class-specific grouped dataset.
- Concept name, e.g. `"red"`, `"triangle"`, `"spots_polka"`: returns dataset restricted to that concept.
- Random dataset key, e.g. `"random500_12"`: returns an additional random sample set with derived seeds.

Location suffixes are supported on queries:

- `"_left"`, `"_right"`, `"_top"`, `"_bot"`
- Example: `"red_top"`, `"random500_12_left"`

## How labels are determined

`class_configs` defines your label space.
Each config entry is a rule matched against every element in an image.
If any element satisfies the rule, the image is marked positive for that class.

Examples:

- `{"shape": "triangle", "color": "red", "texture": None}`:
  any red triangle regardless of texture.
- `{"shape": None, "color": "blue", "texture": "stripes_diagonal"}`:
  any blue diagonally striped object regardless of shape.

Because multiple rules can match in one image, labels are multi-hot.

## `allowed` vs `allowed_combinations`

- `allowed` defines independent value pools for each concept dimension.
- `allowed_combinations` defines exact valid triples.

If `allowed_combinations` is provided, sampled `(shape, color, texture)` triples must appear in that list.
This is the mechanism used in the provided configs to create causal or constrained worlds.

## Examples

## 1) Basic dataset

```python
from elements.classes import ElementDataset

allowed = {
    "shapes": ["square", "circle", "triangle", "plus"],
    "colors": ["red", "green", "blue"],
    "textures": ["solid", "spots_polka", "stripes_diagonal"],
}

class_configs = [
    {"shape": "triangle", "color": "red", "texture": None},
    {"shape": None, "color": "blue", "texture": "stripes_diagonal"},
]

dataset = ElementDataset(
    allowed=allowed,
    class_configs=class_configs,
    n=1000,
    img_size=224,
    element_n=4,
    element_size=64,
    element_size_delta=16,
    element_seed=42,
    loc_seed=123,
)

img_tensor, label = dataset[0]
```

## 2) Restrict to specific concept triples

```python
allowed_combinations = [
    ("triangle", "red", "solid"),
    ("triangle", "red", "spots_polka"),
    ("circle", "blue", "stripes_diagonal"),
]

dataset = ElementDataset(
    allowed=allowed,
    class_configs=class_configs,
    n=1000,
    img_size=224,
    element_n=4,
    element_size=64,
    element_size_delta=16,
    element_seed=42,
    loc_seed=123,
    allowed_combinations=allowed_combinations,
)
```

## 3) Create concept/class/random probe datasets

```python
from elements.classes import ConceptElementDatasetCreator

dataset_creator = ConceptElementDatasetCreator(
    allowed=allowed,
    class_configs=class_configs,
    dataset_kwargs={
        "n": 500,
        "img_size": 224,
        "element_n": 4,
        "element_size": 64,
        "element_size_delta": 16,
        "element_seed": 456,
        "loc_seed": 789,
    },
)

class_ds = dataset_creator("0")
red_ds = dataset_creator("red")
red_top_ds = dataset_creator("red_top")
random_ds = dataset_creator("random500_12")
```

## 4) Use YAML config files from `configs/`

```python
from pathlib import Path
import yaml
from elements.classes import ConceptElementDatasetCreator

with open(Path("configs/simple_dataset.yaml"), "r") as fp:
    config = yaml.safe_load(fp)

params = config["dataset"]["params"]
dataset_creator = ConceptElementDatasetCreator(**params)
dataset = dataset_creator("random500_0")
```

## Practical tips

- Keep `element_size` and `element_n` consistent with `img_size`; too many large objects increase placement failures.
- Set fixed seeds (`element_seed`, `loc_seed`) for reproducibility.
- Use `place_remaining_randomly=False` when location constraints are semantically important for labeling.
- For custom textures, follow the `<family>_<variant>` naming pattern used by `Element`.
- Start from one of the provided configs (`simple_dataset.yaml`, `standard_dataset.yaml`) and modify incrementally.
