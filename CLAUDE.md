# Project context

This is a research project applying **AJIVE** (Angle-based Joint and Individual Variation Explained) to mammography deep learning. Two ResNet classifiers trained on the **VinDR** dataset (cancer BI-RADS and breast density) are decomposed with AJIVE into shared (joint) and task-specific (individual) feature subspaces. `Single_view_AJIVE_heads` then classifies from each subspace separately, and Grad-CAM is used to study what each component attends to.

## File map

| File / dir | Purpose | Status |
|---|---|---|
| `test_vin.py` | Main marimo notebook — VinDR AJIVE analysis, Grad-CAM, localization | **Primary** |
| `train_1vf.py` | Train single-view ResNet on VinDR (CC or MLO) | Active |
| `train_heads.py` | Train AJIVE classification heads | Active |
| `AJIVE_cancer+density.ipynb` | AJIVE: cancer vs density, single view | Reference |
| `AJIVE_CC+MLO_cancer.ipynb` | AJIVE: CC vs MLO views, cancer task | Reference |
| `vinDR_annot.ipynb`, `vinDR_gradCam.ipynb` | Earlier VinDR annotation and saliency analysis | Reference |
| `Elements_AJIVE.ipynb`, `test_elements.py` | Synthetic Elements dataset validation of AJIVE | Secondary |
| `train_elements.py` | Train shape/texture classifiers on Elements | Secondary |
| `synthetic_data.ipynb`, `train_synt.py` | Earlier custom synthetic (blobs+waves, `Synthetic_2v_Dataloader`) | **Abandoned** |
| `src/` | All PyTorch Lightning models and dataloaders |  |
| `python_packages/` | AJIVE (`jive`), PCA (`ya_pca`), mvdr — always add to `sys.path` first |  |

## src module

```python
from src import (
    View_Cancer_Dataloader,       # VinDR, single view (CC or MLO); returns (img, cancer_label, density_label)
    Breast_Cancer_Dataloader,     # VinDR, breast level
    Patient_Cancer_Dataloader,    # VinDR, patient level
    Single_view_model,            # ResNet18; .task=1/2; .generate_grad_cam(x, target_class) → (cam, pred)
    Single_view_AJIVE_heads,      # wraps Single_view_model; heads: joint/individual/both
    Two_view_model,               # CC+MLO two-branch model
    Four_view_two_branch_model,   # bilateral four-view
    Single_Element_Dataloader,    # Elements synthetic; returns (img, shape_label, texture_label)
    Synthetic_2v_Dataloader,      # abandoned old synthetic
)
```

`View_Cancer_Dataloader` normalises images (mean≈108.3, std≈69.6); denorm is needed when displaying with matplotlib. `Single_Element_Dataloader` images are `[0, 1]` — no denorm needed.

## Checkpoint loading — two patterns

**VinDR AJIVE heads** — Lightning `.ckpt`, restored with `load_from_checkpoint` (ajive_model stored in hyperparameters):
```python
ckpt = torch.load(path, weights_only=False)
hp = ckpt["hyper_parameters"]
model = Single_view_AJIVE_heads.load_from_checkpoint(path, model=backbone, ajive_model=hp["ajive_model"], ...)
```

**Elements AJIVE heads** — plain `.pt` dict; AJIVE must be re-fitted from cached features first:
```python
aj = AJIVE(init_signal_ranks=(14, 14), n_jobs=4, center=True)
aj.fit([cache["shape_features"], cache["texture_features"]])  # cache at saved_features/Elements_AJIVE/all_data_train.pt
checkpoint = torch.load(path, weights_only=False)
model = Single_view_AJIVE_heads(model=base, ajive_model=aj, num_class=n,
    active_head=checkpoint["active_head"], hidden_dim=56, drop=0.3, learning_rate=1e-3)
model.load_state_dict(checkpoint["state_dict"])
```

## Key artifacts

| Checkpoint | Model |
|---|---|
| `artifacts/model-5opqyl7e:v0/model.ckpt` | Cancer `Single_view_model` (VinDR) |
| `artifacts/model-kb9kz79s:v0/model.ckpt` | Density `Single_view_model` (VinDR) |
| `artifacts/model-x541jdxj:v0/model.ckpt` | Cancer `Single_view_AJIVE_heads` (VinDR, Lightning) |
| `artifacts/model-ngta50dn:v0/model.ckpt` | Density `Single_view_AJIVE_heads` (VinDR, Lightning) |
| `artifacts/model-908ij4rg:v0/model.ckpt` | Shape `Single_view_model` (Elements) |
| `artifacts/model-qtzv9eop:v0/model.ckpt` | Texture `Single_view_model` (Elements) |
| `artifacts/shape_all_heads.pt` | Shape AJIVE heads (Elements, plain `.pt`) |
| `artifacts/texture_all_heads.pt` | Texture AJIVE heads (Elements, plain `.pt`) |

---

# Marimo notebook assistant

I am a specialized AI assistant designed to help create data science notebooks using marimo. I focus on creating clear, efficient, and reproducible data analysis workflows with marimo's reactive programming model.

If you make edits to the notebook, only edit the contents inside the function decorator with @app.cell.
marimo will automatically handle adding the parameters and return statement of the function. For example,
for each edit, just return:

```
@app.cell
def _():
    <your code here>
    return
```

## Marimo fundamentals

Marimo is a reactive notebook that differs from traditional notebooks in key ways:

- Cells execute automatically when their dependencies change
- Variables cannot be redeclared across cells
- The notebook forms a directed acyclic graph (DAG)
- The last expression in a cell is automatically displayed
- UI elements are reactive and update the notebook automatically

## Code Requirements

1. All code must be complete and runnable
2. Follow consistent coding style throughout
3. Include descriptive variable names and helpful comments
4. Import all modules in the first cell, always including `import marimo as mo`
5. Never redeclare variables across cells
6. Ensure no cycles in notebook dependency graph
7. The last expression in a cell is automatically displayed, just like in Jupyter notebooks.
8. Don't include comments in markdown cells
9. Don't include comments in SQL cells
10. Never define anything using `global`.

## Reactivity

Marimo's reactivity means:

- When a variable changes, all cells that use that variable automatically re-execute
- UI elements trigger updates when their values change without explicit callbacks
- UI element values are accessed through `.value` attribute
- You cannot access a UI element's value in the same cell where it's defined
- Cells prefixed with an underscore (e.g. _my_var) are local to the cell and cannot be accessed by other cells

## Best Practices

<data_handling>

- Use polars for data manipulation
- Implement proper data validation
- Handle missing values appropriately
- Use efficient data structures
- A variable in the last expression of a cell is automatically displayed as a table
</data_handling>

<visualization>
- For matplotlib: use plt.gca() as the last expression instead of plt.show()
- For plotly: return the figure object directly
- For altair: return the chart object directly. Add tooltips where appropriate. You can pass polars dataframes directly to altair.
- Include proper labels, titles, and color schemes
- Make visualizations interactive where appropriate
</visualization>

<ui_elements>

- Access UI element values with .value attribute (e.g., slider.value)
- Create UI elements in one cell and reference them in later cells
- Create intuitive layouts with mo.hstack(), mo.vstack(), and mo.tabs()
- Prefer reactive updates over callbacks (marimo handles reactivity automatically)
- Group related UI elements for better organization
</ui_elements>

<sql>
- When writing duckdb, prefer using marimo's SQL cells, which start with df = mo.sql(f"""<your query>""") for DuckDB, or df = mo.sql(f"""<your query>""", engine=engine) for other SQL engines.
- See the SQL with duckdb example for an example on how to do this
- Don't add comments in cells that use mo.sql()
</sql>

## Troubleshooting

Common issues and solutions:

- Circular dependencies: Reorganize code to remove cycles in the dependency graph
- UI element value access: Move access to a separate cell from definition
- Visualization not showing: Ensure the visualization object is the last expression

After generating a notebook, run `marimo check --fix` to catch and
automatically resolve common formatting issues, and detect common pitfalls.

## Available UI elements

- `mo.ui.altair_chart(altair_chart)`
- `mo.ui.button(value=None, kind='primary')`
- `mo.ui.run_button(label=None, tooltip=None, kind='primary')`
- `mo.ui.checkbox(label='', value=False)`
- `mo.ui.date(value=None, label=None, full_width=False)`
- `mo.ui.dropdown(options, value=None, label=None, full_width=False)`
- `mo.ui.file(label='', multiple=False, full_width=False)`
- `mo.ui.number(value=None, label=None, full_width=False)`
- `mo.ui.radio(options, value=None, label=None, full_width=False)`
- `mo.ui.refresh(options: List[str], default_interval: str)`
- `mo.ui.slider(start, stop, value=None, label=None, full_width=False, step=None)`
- `mo.ui.range_slider(start, stop, value=None, label=None, full_width=False, step=None)`
- `mo.ui.table(data, columns=None, on_select=None, sortable=True, filterable=True)`
- `mo.ui.text(value='', label=None, full_width=False)`
- `mo.ui.text_area(value='', label=None, full_width=False)`
- `mo.ui.data_explorer(df)`
- `mo.ui.dataframe(df)`
- `mo.ui.plotly(plotly_figure)`
- `mo.ui.tabs(elements: dict[str, mo.ui.Element])`
- `mo.ui.array(elements: list[mo.ui.Element])`
- `mo.ui.form(element: mo.ui.Element, label='', bordered=True)`

## Layout and utility functions

- `mo.md(text)` - display markdown
- `mo.stop(predicate, output=None)` - stop execution conditionally
- `mo.output.append(value)` - append to the output when it is not the last expression
- `mo.output.replace(value)` - replace the output when it is not the last expression
- `mo.Html(html)` - display HTML
- `mo.image(image)` - display an image
- `mo.hstack(elements)` - stack elements horizontally
- `mo.vstack(elements)` - stack elements vertically
- `mo.tabs(elements)` - create a tabbed interface

## Examples

<example title="Markdown ccell">
```
@app.cell
def _():
    mo.md("""
    # Hello world
    This is a _markdown_ **cell**.
    """)
    return
```
</example>

<example title="Basic UI with reactivity">
```
@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    import numpy as np
    return

@app.cell
def _():
    n_points = mo.ui.slider(10, 100, value=50, label="Number of points")
    n_points
    return

@app.cell
def _():
    x = np.random.rand(n_points.value)
    y = np.random.rand(n_points.value)

    df = pl.DataFrame({"x": x, "y": y})

    chart = alt.Chart(df).mark_circle(opacity=0.7).encode(
        x=alt.X('x', title='X axis'),
        y=alt.Y('y', title='Y axis')
    ).properties(
        title=f"Scatter plot with {n_points.value} points",
        width=400,
        height=300
    )

    chart
    return

```
</example>

<example title="Multiple UI elements">
```

@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    return

@app.cell
def _():
    iris = pl.read_csv("hf://datasets/scikit-learn/iris/Iris.csv")
    return

@app.cell
def _():
    species_selector = mo.ui.dropdown(
        options=["All"] + iris["Species"].unique().to_list(),
        value="All",
        label="Species",
    )
    x_feature = mo.ui.dropdown(
        options=iris.select(pl.col(pl.Float64, pl.Int64)).columns,
        value="SepalLengthCm",
        label="X Feature",
    )
    y_feature = mo.ui.dropdown(
        options=iris.select(pl.col(pl.Float64, pl.Int64)).columns,
        value="SepalWidthCm",
        label="Y Feature",
    )
    mo.hstack([species_selector, x_feature, y_feature])
    return

@app.cell
def _():
    filtered_data = iris if species_selector.value == "All" else iris.filter(pl.col("Species") == species_selector.value)

    chart = alt.Chart(filtered_data).mark_circle().encode(
        x=alt.X(x_feature.value, title=x_feature.value),
        y=alt.Y(y_feature.value, title=y_feature.value),
        color='Species'
    ).properties(
        title=f"{y_feature.value} vs {x_feature.value}",
        width=500,
        height=400
    )

    chart
    return

```
</example>

<example title="Conditional Outputs">
```

@app.cell
def _():
    mo.stop(not data.value, mo.md("No data to display"))

    if mode.value == "scatter":
        mo.output.replace(render_scatter(data.value))
    else:
        mo.output.replace(render_bar_chart(data.value))
    return

```
</example>

