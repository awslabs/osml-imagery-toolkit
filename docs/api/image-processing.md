# aws.osml.image\_processing

Pixel-level operations on overhead imagery. This package covers reading and stitching
blocks from tiled sources, resampling between resolutions, display range adjustment
(DRA), SAR complex-to-display remap, image pyramid construction, warping and
orthorectification, chipping with associated metadata, and map tile generation. It does
**not** include sensor model math (see `photogrammetry`), metadata/TRE parsing (see
`metadata`), or feature geolocation (see `features`).

### Package dependencies

`image_processing` imports `photogrammetry` for sensor model interfaces used in
orthorectification and warping operations. It imports `formats` for SICD/SIDD XML
updaters that keep chip metadata consistent after sub-setting. It does not depend on
`metadata`, `elevation`, or `features`.

### Design abstractions

**ImageAssetProvider** (duck-typed protocol) is the universal source contract. Any
object that exposes `get_block(row, col, resolution_level, bands)`, `num_rows`,
`num_columns`, `num_bands`, `pixel_value_type`, `block_grid_size`, and `metadata`
satisfies the protocol. All adapters produce and consume this interface so they compose
freely.

**Adapter/Decorator pattern.** Providers wrap other providers to layer behavior —
caching, retiling, downsampling, applying a function, or warping. Pipelines are built by
stacking adapters; no inheritance hierarchy is required.

**ProcessingChain** is a composable sequence of `ndarray -> ndarray` callables with
output metadata. Chains are callable, nestable via `compose()`, and applied to pixels
after reading.

**TiledImagePyramid** groups providers into a multi-resolution stack (level 0 = full
resolution). It is the read-side entry point for resolution-aware access.

**GridBuilder** (ABC) defines how output tile grids map back to source pixel
coordinates.

**Block-oriented I/O.** All pixel access flows through `get_block()`. Random-access
windows are assembled from blocks internally. Arrays are always **CHW** (channels,
height, width).

### Adapter composition

```{mermaid}
flowchart LR
    A[Source Asset] --> B[RetiledProvider]
    B --> C[CachedProvider]
    C --> D[MappedProvider]
    D --> E[TiledImagePyramid]
    E --> F[WarpedImageProvider]
    F --> G[ChipFactory]
```

### Display pipeline

```{mermaid}
flowchart LR
    A[raw CHW pixels] --> B[ProcessingChain]
    B --> C[band_select]
    C --> D["DRA / LUT"]
    D --> E[uint8 display output]
```

### Contributor rules

New pixel operations should be implemented as an `ndarray -> ndarray` function usable as
a ProcessingChain step, or as a new ImageAssetProvider adapter wrapping an existing
provider. New output formats require a `ChipMetadataBuilder` subclass. Maintain the
duck-typed provider protocol — do not introduce a base class import requirement.

## Chip Factory

```{eval-rst}
.. autoclass:: aws.osml.image_processing.ChipFactory
   :members:
   :undoc-members:
   :show-inheritance:
```

## Map Tile Sets

```{eval-rst}
.. autoclass:: aws.osml.image_processing.MapTileSetFactory
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.image_processing.WellKnownMapTileSet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.image_processing.MapTileSet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.image_processing.MapTile
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.image_processing.MapTileId
   :members:
   :undoc-members:
   :show-inheritance:
```

## Complex Imagery Remap

```{eval-rst}
.. autofunction:: aws.osml.image_processing.is_complex

.. autoclass:: aws.osml.image_processing.ComplexRemapFactory
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: aws.osml.image_processing.load_complex_remap

.. autofunction:: aws.osml.image_processing.quarter_power_remap

.. autofunction:: aws.osml.image_processing.magnitude_remap

.. autofunction:: aws.osml.image_processing.decode_to_iq

.. autofunction:: aws.osml.image_processing.complex_to_power

.. autofunction:: aws.osml.image_processing.power_to_decibels
```

## SICD/SIDD Updaters

```{eval-rst}
.. automodule:: aws.osml.image_processing.sicd_updater
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aws.osml.image_processing.sidd_updater
   :members:
   :undoc-members:
   :show-inheritance:
```

## Resampling Functions

```{eval-rst}
.. automodule:: aws.osml.image_processing.resample
   :members:
   :undoc-members:
   :show-inheritance:
```

## SIPS RRDS Resampler

```{eval-rst}
.. automodule:: aws.osml.image_processing.sips_resample
   :members:
   :undoc-members:
   :show-inheritance:
```

## Pyramid Builder

```{eval-rst}
.. autoclass:: aws.osml.image_processing.PyramidBuilder
   :members:
   :undoc-members:
   :show-inheritance:
```

## Downsampled Image Provider

```{eval-rst}
.. autoclass:: aws.osml.image_processing.DownsampledImageProvider
   :members:
   :undoc-members:
   :show-inheritance:
```

## Tiled Image Pyramid

```{eval-rst}
.. autoclass:: aws.osml.image_processing.TiledImagePyramid
   :members:
   :undoc-members:
   :show-inheritance:
```

## Pyramid Helpers

```{eval-rst}
.. autofunction:: aws.osml.image_processing.iter_blocks

.. autofunction:: aws.osml.image_processing.build_pyramid_levels
```
