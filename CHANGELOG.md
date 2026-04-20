# Changelog

## 1.0.0 — 2026-04-15

Complete rewrite of the dice-plot engine and public API, plus a full domino
API refactor to match the new 1.0 conventions.

### Breaking changes

- **`dots` renamed to `pips`** throughout. The `dots` parameter, `dot_colors`,
  and related names are gone. Use `pips`, `pip_colors`, `pip_scale`,
  `pips_label`, `pips_order`, `max_pips` instead. ("Pip" is the correct term
  for the marks on a die face.)
- **Seaborn-style function signature.** `dice_plot(data, x, y, pips, ...)`
  now takes positional `data`, `x`, `y`, `pips` followed by keyword-only
  options. The old dict-based / mixed-positional calling convention is removed.
- **Native return types.** matplotlib: `(Figure, Axes)` when creating a new
  figure, just `Axes` when the caller supplies `ax=`. plotly: `go.Figure`.
  The old wrapper return types are gone.
- **`domino_plot` rewritten around a column-first API.** The old
  `gene_list`/`var_id`/`logfc_col`/`pval_col`-style entry point is replaced by
  `domino_plot(data, feature, celltype, contrast, *, fill=..., size=..., ...)`
  with native backend returns, explicit contrast ordering, and backend-specific
  `ax=` / `fig=` composition hooks.
- **`n=6` uses traditional die-face layout** (two vertical columns) instead
  of the transposed two-row layout from earlier versions. This is an
  intentional divergence from `ggdiceplot::make_offsets`.

### New features

- **Up to 9 pips.** The 3×3 sub-grid now supports `max_pips=9` with
  traditional die-face lookup for every value 1–9.
- **Grid geometry ported from kuva.** `_layout.py` replaces the old ad-hoc
  positioning code with a direct port of `kuva/src/plot/diceplot.rs`, ensuring
  matplotlib and plotly produce identical pip positions and tile sizes.
- **Per-pip continuous fill and size.** Pass `fill="col"` and/or `size="col"`
  for numeric columns — each pip gets its own colour (colorbar) and radius
  (size legend). Works with both backends.
- **Per-pip discrete fill.** Pass `fill="col"` + `fill_palette={val: hex}`
  for discrete colour encoding per pip slot.
- **Legend stack.** A right-side legend panel stacks position legend, colorbar,
  and size legend, matching `ggdiceplot::draw_key` semantics. Skipped when
  the caller provides `ax=` / `fig=`.
- **`ax=` / `fig=` composability.** Draw into an existing matplotlib `Axes`
  or plotly `Figure` to build multi-panel layouts.
- **Domino preprocessing rewrite.** Domino plots now validate their structural
  columns up front, enforce exactly two contrast slots, compute shared backend
  geometry once, and use the same color/size range semantics as `dice_plot`.
- **Tile geometry controls.** `tile_width`, `tile_height`, `pip_scale`,
  `grid_lines` parameters.
- **Label controls.** `fill_label`, `size_label`, `pips_label` set legend
  titles; `xlabel`, `ylabel`, `title` set axis / figure titles.
- **pixi environment.** `pixi.toml` with `test`, `example`, `build`, `check`
  tasks. The package is installed as an editable pypi dependency.

### Example scripts

- 1-to-1 ports of ggdiceplot demo plots: `oral_microbiome.py`,
  `oral_microbiome_fill_only.py`, `mirna_direction.py`, `zebra_domino.py`.
- Standalone `example_domino.py`: demonstrates the refactored domino API.
- Creative `pathways_nine.py`: nine signaling pathways on a 3×3 die face.
- Showcase `example.py`: generates all `images/dice_*.png` gallery images.

### Tests

- `test_dice_plot.py`: smoke tests covering all three modes (categorical,
  continuous, discrete fill), both backends, return types, and preprocessing.
- `test_domino_plot.py`: preprocessing, switch-axis, composition, validation,
  and backend smoke tests for the refactored domino API.
- `test_layout.py`: geometry unit tests for pip positions and tile sizing.

## 0.0.2 — 2025

Initial PyPI release with basic dice and domino plot support.
