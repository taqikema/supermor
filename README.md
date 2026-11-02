# SUPERMOR

Multi-frequency three-dimensional controlled-source electromagnetic (CSEM) modelling with gradient-field splitting and inexact shifted solves.

## Overview

This repository accompanies the SUPERMOR study and provides the solver extension, mesh generators, numerical experiments and plotting scripts for three models:

| Numerical example | Experiments | Paper figures |
| --- | --- | --- |
| Shifted conductive block | Independent cross-validation, gradient-field splitting, single- and dual-pole comparisons | 1–6 |
| Layered background with a spherical conductor | MPI scaling and frequency-count comparisons | 7–10 |
| MR3D anisotropic model | Five mesh levels and four-method comparisons | 11–12 |

The four methods are **DFOM** (direct full-order), **IFOM** (iterative full-order), **RK** (restarted rational Krylov) and **SUPERMOR**. Internal result labels `CPC` and `SUMOR` denote RK and SUPERMOR, respectively. The paper workflows use P1 edge elements. The core also contains selected P2 extensions; their supported combinations are described in [ENVIRONMENT.txt](ENVIRONMENT.txt).

SUPERMOR extends a separately installed custEM framework. The numerical source versions are recorded in [VERSION.txt](VERSION.txt), and [FILE_FIGURE_MAP.csv](FILE_FIGURE_MAP.csv) maps the paper figures to their scripts and inputs.

## Repository structure

```text
supermor/
├── core/supermor/       Solver implementations and finite-element operators
├── mesh_gen/            Paper mesh-generation workflows
├── runs/                Numerical case entries and shared runtime utilities
├── figs/                Model and result plotting scripts
├── data/
│   ├── mesh_data/       Geometry inputs and locally generated meshes
│   ├── model_data/      Prepared MR3D material arrays
│   ├── raw_data/        Original MR3D SEG-Y volumes
│   └── figure_data/     Model geometry and the figure-7 display scene
├── outputs/             Numerical results, paper plotting inputs and figures
├── tests/               Core, mesh, run-IO and plotting tests
├── environment.yml      Numerical environment specification
└── README.md            Installation and reproduction guide
```

Paths resolve relative to this checkout. Numerical inputs belong under `data/`; new results and the figures produced by the commands below belong under `outputs/`. The downloaded `outputs/.../paper/` trees hold the paper's plotting inputs, while new computations use separate run directories.

## Requirements and installation

### Numerical environment

The supplied Conda environment targets **Linux x86-64**, including WSL2. Key recorded numerical versions are:

| Component | Recorded version |
| --- | --- |
| Python | 3.11.9 |
| Legacy FEniCS / DOLFIN | 2019.1.0 |
| PETSc, real scalar build with HYPRE/AMS and MUMPS | 3.20.6 |
| MPICH / mpi4py | 4.2.3 / 3.1.6 |
| NumPy / SciPy | 1.26.4 / 1.17.1 |
| pyGIMLi / TetGen | 1.5.0 / 1.6.0 |

Download or clone this repository, enter its root directory, and create the environment:

```bash
conda env create -f environment.yml
conda activate supermor
python -m pip install --no-deps --no-build-isolation .
```

Use the activated environment's Python and MPI launcher. The environment specification pins the official custEM and COMET dependencies. [ENVIRONMENT.txt](ENVIRONMENT.txt) provides the full version and platform notes; `requirements.txt` alone does not supply the FEniCS/PETSc/MPI stack.

Use one numerical thread per MPI rank for the paper commands:

```bash
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/supermor-cache"
```

### Plotting and MR3D preprocessing

Matplotlib, Pillow and VTK are included in the numerical environment. Figure 11 and MR3D material preparation additionally use SEG-Y input through `segyio`:

```bash
python -m pip install --no-deps segyio==1.9.14
```

The paper style uses **Times New Roman**, except for figures 2–4, which retain DejaVu Sans. Install the Times font on your system or point to a directory containing its `times*.ttf` files:

```bash
export SUMOR_PAPER_FONT_DIR=/path/to/times-fonts
```

Figure 7 uses VTK offscreen rendering and requires a working EGL/OpenGL backend. Font files are not distributed with the code.

### Additional solver for block cross-validation

The independent emg3d comparison requires **emg3d v1.9.1** from the clean Git commit `0624e3c9779503793d271dd940daca59e2a3fa41`. A package installation without that Git checkout does not provide the source identity required by the worker.

The following setup uses the active environment and keeps the external checkout beside this repository:

```bash
git clone --branch v1.9.1 https://github.com/emsig/emg3d.git ../emg3d-paper
git -C ../emg3d-paper checkout 0624e3c9779503793d271dd940daca59e2a3fa41
python -m pip install ../emg3d-paper threadpoolctl
export EMG3D_SOURCE="$(cd ../emg3d-paper && pwd)"
export EMG3D_PYTHON="$(command -v python)"
```

An existing emg3d environment may be used instead: set `EMG3D_PYTHON` to its Python executable and `EMG3D_SOURCE` to the matching clean checkout. The other numerical examples do not require emg3d.

`threadpoolctl` is used by the cross-validation worker to inspect the active native thread pools.

## Data preparation

### SUPERMOR basic data package

Download **`supermor_data.zip`** from the [SUPERMOR basic dataset on Zenodo](https://doi.org/10.5281/zenodo.23876222) (DOI: **10.5281/zenodo.23876222**). The package contains the following basic inputs:

| Contents | Files | Uncompressed size | Destination after extraction |
| --- | ---: | ---: | --- |
| Paper plotting inputs: CSV, NPZ and JSON in five experiment trees | 21 | 14.67 MiB | `outputs/<case>/<experiment>/paper/` |
| Complete figure-7 VTK scene | 8 | 0.56 MiB | `data/figure_data/case2_layered_sphere/geometry/` |
| Six processed MR3D interface XYZ files | 6 | 71.09 MiB | `data/mesh_data/case3_mr3d/prepared/interfaces/` |
| **Total** | **35** | **86.32 MiB** | |

The exact total is **90,513,147 bytes before compression**; it is not the ZIP download size. From the repository root, extract the downloaded archive while preserving its `data/` and `outputs/` paths:

```bash
unzip /path/to/supermor_data.zip -d .
```

The five paper-input trees are:

```text
outputs/case1_block/cross_validation/paper/
outputs/case1_block/strategy/paper/          # gradient/ and poles/
outputs/case2_layered_sphere/mpi_scaling/paper/
outputs/case2_layered_sphere/frequency_count/paper/
outputs/case3_mr3d/mesh_scaling/paper/
```

Extract directly into the checkout, rather than adding a `supermor_data/` directory above these paths. Keep `survey_scene.vtm` and its seven referenced VTU/VTP files in their original relative layout. This scene displays the model in figure 7; it is separate from the numerical sphere mesh.

Final computational meshes and frozen reference responses are not included in the package. Generate each required mesh using the commands below; the numerical workflows compute their reference solutions. Intermediate meshes, full fields, matrices, LU factors, Krylov bases, caches and rendered images are also outside the basic package.

### Small inputs included with Git

The following four inputs are already supplied with the source and are not duplicated in the basic package:

- `data/figure_data/case1_block/geometry/model_parameters.json`
- `data/mesh_data/case2_layered_sphere/inputs/sphere_surface_template.npz`
- `data/mesh_data/case3_mr3d/prepared/manifest.json`
- `data/mesh_data/case3_mr3d/prepared/survey.npz`

### Original MR3D resistivity volumes

For MR3D calculations and figure 11, obtain the following files from the [original MR3D model by Carvalho and Menezes (2017)](https://doi.org/10.5281/zenodo.400233) and place them here:

```text
data/raw_data/case3_mr3d/
├── Horizontal_resistivity.sgy
└── Vertical_Resistivity.sgy
```

Together they occupy approximately **3.64 GiB**. They remain in the original data record and are not duplicated in `supermor_data.zip`. Use the original model rather than its separately published extended version.

Before MR3D forward modelling, the material-preparation command in Case 3 creates:

```text
data/model_data/case3_mr3d/
├── manifest.json
├── axes.npz
├── log_rho_h.npy
└── log_rho_v.npy
```

These arrays are regenerated from the SEG-Y volumes, independently of mesh generation. Figure 11 reads the same original volumes directly, so no additional raw-model copy is needed.

## Reproducing the numerical examples

Run the following commands from the repository root after installing the environment and preparing the inputs. Each example follows **input preparation → mesh generation → numerical computation → plotting**.

Choose a new, empty output directory for each numerical campaign; replace `run_01` when repeating an experiment. Coordinators launch their MPI workers themselves. Start the coordinator once within an allocation containing the required resources, rather than wrapping it in another `mpiexec` invocation.

For mesh generation, `--data-root` replaces `data/mesh_data/`. For numerical runs, it replaces the complete `data/` directory. The commands below use the default layout.

### Case 1: shifted conductive block

#### Independent cross-validation — figures 1–4

This comparison uses the Git-supplied geometry definitions and generates both the finite-element mesh and the fixed emg3d mesh inputs:

```bash
python -B mesh_gen/generate.py block_cross
python -B mesh_gen/generate.py block_emg3d

python -B runs/case1_block/cross_validation/run_cross_validation.py \
  --emg3d-source "$EMG3D_SOURCE" \
  --emg3d-python "$EMG3D_PYTHON" \
  --output outputs/case1_block/cross_validation/run_01

python -B figs/case1_block/plot_sumor_block_mesh_section.py \
  --output-dir outputs/figures/case1_block

python -B figs/case1_block/plot_sumor_block_cross_validation.py \
  --run-dir outputs/case1_block/cross_validation/run_01 \
  --output-dir outputs/figures/case1_block
```

The coordinator runs DFOM with 16 MPI ranks, SUPERMOR with 32 MPI ranks and emg3d as a single-process solver. The result plot consumes the complete cross-validation run directory.

#### Gradient splitting and pole comparisons — figures 5–6

Generate the separate strategy mesh, then run the five configurations and their independent reference:

```bash
python -B mesh_gen/generate.py block_strategy

python -B runs/case1_block/strategy/run_strategy.py \
  --strategy all --compute-reference --save-fields \
  --output outputs/case1_block/strategy/run_01

python -B figs/case1_block/plot_sumor_hpc_formulation_costs_errors.py \
  --run-dir outputs/case1_block/strategy/run_01 \
  --output-dir outputs/figures/case1_block

python -B figs/case1_block/plot_sumor_hpc_pole_costs_errors.py \
  --run-dir outputs/case1_block/strategy/run_01 \
  --output-dir outputs/figures/case1_block
```

Both plotting scripts take the same campaign root, which contains the reference and all strategy results. A single dual-pole computation can be selected with `--strategy dual_pole`; the full figure workflows above use `--strategy all`.

### Case 2: layered background with a spherical conductor

The sphere template is included with Git. Extract the figure-7 display scene from the basic package before rendering that model figure.

```bash
python -B mesh_gen/generate.py sphere

python -B runs/case2_layered_sphere/run_mpi_scaling.py \
  --output outputs/case2_layered_sphere/mpi_scaling/run_01

python -B runs/case2_layered_sphere/run_frequency_count.py \
  --output outputs/case2_layered_sphere/frequency_count/run_01

python -B figs/case2_layered_sphere/plot_sumor_layered_sphere_survey.py \
  --output-dir outputs/figures/case2_layered_sphere

python -B figs/case2_layered_sphere/plot_supermor_paper_mpi_performance.py \
  --run-dir outputs/case2_layered_sphere/mpi_scaling/run_01 \
  --output-dir outputs/figures/case2_layered_sphere

python -B figs/case2_layered_sphere/plot_sumor_frequency_count_four_methods.py \
  --run-dir outputs/case2_layered_sphere/frequency_count/run_01 \
  --output-dir outputs/figures/case2_layered_sphere
```

The MPI campaign compares four methods with 1, 4, 8, 16, 32 and 64 ranks. The frequency-count campaign uses 8 ranks and seven frequency counts. Their complete scans supply figures 8 and 9–10, respectively.

The MPI SUPERMOR entry defaults to `paper_mpi`, retaining the historical P1 kernel used for that experiment. The frequency-count entry defaults to `current`. These selections are already made by the supplied commands.

### Case 3: MR3D anisotropic model

Extract the six processed interfaces from the basic package and download the two original SEG-Y volumes as described above. Prepare the material arrays and generate the five mesh levels:

```bash
python -B runs/case3_mr3d/prepare_materials.py \
  --raw-dir data/raw_data/case3_mr3d

python -B mesh_gen/generate.py mr3d

python -B runs/case3_mr3d/run_mesh_scaling.py \
  --output outputs/case3_mr3d/mesh_scaling/run_01

python -B figs/case3_mr3d/plot_mr3d_section.py \
  --output-dir outputs/figures/case3_mr3d

python -B figs/case3_mr3d/plot_sumor_mr3d_scaling_four_methods.py \
  --run-dir outputs/case3_mr3d/mesh_scaling/run_01 \
  --output-dir outputs/figures/case3_mr3d
```

The preparation entry also offers `--fetch` to download the two original volumes explicitly. The six mesh interfaces are supplied separately by the basic package.

The default scan uses 8 MPI ranks and the paper's 17 configurations: SUPERMOR and IFOM on M1–M5 with three repeats, RK on M1–M5 once, and DFOM on M1–M2 once. DFOM M3–M5 require an explicit opt-in. Material preprocessing and numerical mesh generation are separate required steps.

### MPI launcher options

Use your allocation's launcher and binding settings. If MPICH on WSL requires an explicit local fork launcher, the coordinators accept these overrides:

| Entry | Additional argument |
| --- | --- |
| Block cross-validation | `--mpiexec "mpiexec -launcher fork -bind-to core"` |
| Sphere MPI or frequency-count scan | `--launcher-args="-launcher fork -bind-to core"` |
| MR3D mesh scan | `--mpi-launcher "mpiexec -launcher fork -bind-to core"` |

The block strategy comparison fixes one MPI rank. Use each entry's `--help` for its available options. Mesh generation and campaign coordinators also provide `--dry-run` command previews.

## Computational resources

Selected measurements from the paper are shown below. They are per-run references, not total campaign times or minimum hardware requirements. Memory is aggregate sampled PSS unless explicitly labelled RSS; 1 GiB is 2^30 bytes.

| Experiment | CPU | MPI ranks × threads/rank | Measured peak memory | Measured time |
| --- | --- | ---: | ---: | ---: |
| Block cross-validation, DFOM, four frequencies | Model not recorded in retained summary | 16 × 1 | 535.42 GiB RSS | 11,178.61 s |
| Block dual-pole SUPERMOR | Model not recorded in retained summary | 1 × 1 | 1.72 GiB PSS | 178.59 s |
| Sphere, SUPERMOR, 20 frequencies | Xeon Gold 6248 @ 2.50 GHz | 16 × 1 | 16.65 GiB PSS | 240.33 s |
| Sphere, SUPERMOR, 40 frequencies | Xeon Gold 6248 @ 2.50 GHz | 8 × 1 | 15.36 GiB PSS | 380.80 s |
| MR3D, SUPERMOR, M1–M5 | Xeon Gold 6248 @ 2.50 GHz | 8 × 1 | 5.36–39.08 GiB PSS | 19.68–413.43 s |
| MR3D, RK, M5 | Xeon Gold 6248 @ 2.50 GHz | 8 × 1 | 547.66 GiB PSS | 7,133.74 s |

The MR3D SUPERMOR endpoints correspond to M1 and M5: time is the median of three runs and memory is their maximum. Other methods can require substantially more memory, as the RK row illustrates. Runtime and memory depend on the machine and parallel configuration. Full measurement inputs are in the basic package's `outputs/.../paper/` trees; the cross-validation DFOM reference is also summarized in the associated manuscript records.

## License and support

Source licensing terms are provided in [LICENSE.rst](LICENSE.rst). [THIRD_PARTY.txt](THIRD_PARTY.txt) records upstream software and model attribution. For the basic dataset's license, see the Zenodo record linked above.

For questions and bug reports, open an issue in this repository with your platform, dependency versions, command and relevant log excerpt.
