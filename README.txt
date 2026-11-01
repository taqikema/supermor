SUPERMOR paper reproduction package
==================================

STATUS
------
The independent supermor package is in core/supermor; numerical tests are
in tests/. It extends a separately installed, unmodified official custEM.
The numerical environment and pinned upstream sources are recorded in
root environment.yml; installation commands are provided below.
CORE_VALIDATION.json records the completed migration checks and their limits.
mesh_data/, runs/ and figs/ remain release destinations to populate; their
script/data names below are planned locations, not implemented case entries.
The complete paper release is not yet assembled. Source-to-paper mapping,
mesh provenance, dependency notes and the original license are provided.
This root README is the single usage guide for all directories below.

SCOPE
-----
The intended release provides:
  1. Code, configuration and necessary inputs to rerun the paper's cases.
  2. Plotting scripts and their frozen inputs to redraw the published
     figures without rerunning the large numerical simulations.

Solver outputs that the runs scripts regenerate do not all need a second
archived copy. Complete raw logs, full fields, matrices, LU factors and
Krylov bases are not required contents of this plan. Keep published timing
and memory values in the plotting inputs, with units and aggregation rules.

For each case, supply either the final usable mesh and companion runtime
inputs, or a complete tested mesh-generation workflow and all its inputs.
Both alternatives are not required. If a runtime still imports validation
helpers from a mesh generator, retain or extract those helpers before
omitting the generator itself.

RELEASE TREE (CORE POPULATED; CASE AND FIGURE ENTRIES PLANNED)
------------------------------------------------------------
supermor/
|-- README.txt
|-- pyproject.toml                   SUPERMOR package/version and upstream pin
|-- requirements.txt                 Candidate plotting dependencies
|-- environment.yml                 Version/build-pinned linux-64 environment
|-- CORE_VALIDATION.json             Migration results and acceptance limits
|-- ENVIRONMENT.txt                  Numerical stack, exact versions and commands
|-- LICENSE.rst                      Existing custEM LGPLv3/GPL license text
|-- THIRD_PARTY.txt                  Input sources and existing attribution
|-- VERSION.txt                      Source commit; add final release tag/DOI
|-- FILE_FIGURE_MAP.csv               Paper item -> source -> proposed release path
|-- MESH_AUDIT_20261101.json          Local source/artifact/parameter audit evidence
|-- core/
|   `-- supermor/
|       |-- __init__.py              Independent public API
|       |-- methods.py               Operator/model solver interfaces
|       |-- full_order.py            Production MPI DFOM/IFOM
|       |-- mor.py                   RK/SUPERMOR numerical kernels
|       |-- operators.py             P1/P2 assembly, line source, E/H sampling
|       |-- materials.py             Native distributed DG0 VTI assignment
|       `-- mor_high_order.py        P2 exact-sequence/AMS transfers
|-- tests/                           Algebra, P1/P2, official H5 and A/B checks
|-- mesh_data/
|   |-- case1_block/
|   |   |-- cross_validation/
|   |   |   |-- custem/              Final fine H5, domain markers, companion inputs
|   |   |   `-- emg3d/               Frozen axes/material grid and required metadata
|   |   `-- strategy/                Separate mesh for gradient/pole comparisons
|   |-- case2_layered_sphere/        Formal H5, parameters and receivers.csv
|   `-- case3_mr3d/
|       |-- material/                Axes, log_rho_h/log_rho_v and survey inputs
|       |-- M1/                      Formal mesh plus companion parameters
|       |-- M2/
|       |-- M3/
|       |-- M4/
|       `-- M5/
|-- runs/
|   |-- common/                     Shared loading, sampling and monitoring helpers
|   |-- case1_block/
|   |   |-- run_cross_validation.py
|   |   |-- run_gradient.py
|   |   `-- run_poles.py
|   |-- case2_layered_sphere/
|   |   |-- run_mpi_scaling.py
|   |   `-- run_frequency_count.py
|   `-- case3_mr3d/
|       `-- run_mesh_scaling.py
|-- figs/
|   |-- plot_all.py                  Proposed unified redraw entry
|   |-- common/                     Shared style, survey renderer and method styles
|   |-- case1_block/
|   |   |-- plot_sumor_block_mesh_section.py
|   |   |-- plot_sumor_block_cross_validation.py
|   |   |-- plot_sumor_hpc_formulation_costs_errors.py
|   |   |-- plot_sumor_hpc_pole_costs_errors.py
|   |   `-- data/
|   |       |-- geometry/           Fig. 1: frozen model parameters (JSON)
|   |       |-- cross_validation/   Figs. 2-4, Table 3: CSV/NPZ drawing inputs
|   |       |-- gradient/           Fig. 5: comparison and frequency metrics
|   |       `-- poles/              Fig. 6: comparison and frequency metrics
|   |-- case2_layered_sphere/
|   |   |-- plot_sumor_layered_sphere_survey.py
|   |   |-- plot_supermor_paper_mpi_performance.py
|   |   |-- plot_sumor_frequency_count_four_methods.py
|   |   `-- data/
|   |       |-- geometry/           Fig. 7: VTM and every referenced VTU/VTP
|   |       |-- mpi_scaling/        Fig. 8: measured JSON/CSV
|   |       `-- frequency_count/    Figs. 9-10: current 28-run summary CSV
|   `-- case3_mr3d/
|       |-- plot_mr3d_section.py     Proposed extraction of the Fig. 11 renderer
|       |-- plot_sumor_mr3d_scaling_four_methods.py
|       `-- data/
|           |-- section/            Fig. 11: frozen section arrays or source inputs
|           `-- scaling/            Fig. 12, Tables 4-5: current performance.csv
`-- outputs/                         Generated locally; not bundled input data
    |-- runs/
    `-- figures/

RELEASE CONTENT NOTES
---------------------
core/
  Seven Python files provide the paper extension. No custEM source tree,
  upstream VERSION.rst, modified MOD/Solver classes, examples or binaries
  are bundled. Official custEM commit
  be0ef8729761a7eae32ababc8b362514d84f5f0a is an external dependency.
  The project's own version is defined only in pyproject.toml.

  Public interfaces:
    from supermor import assemble_mor, assign_native_vti, solve, solve_model
    solve(ops, frequencies, method='SUPERMOR', emit=None, **options)
    solve_model(model, method='SUPERMOR', receiver_domain=None, **options)

  solve accepts P1 DFOM/IFOM/RK/SUPERMOR; CPC/SUMOR alias RK/SUPERMOR.
  Defaults, production tolerances, PETSc prefixes and native statistics are
  retained. P2 operator solves support DFOM/RK/SUPERMOR; P2 IFOM is explicitly
  unsupported. P2 SUPERMOR supports residual_mr without residual correction,
  with split_u or total_field. P2 chain/correction modes remain unsupported.
  P2 model solves retain the existing SUPERMOR-only scope.

  Create the official MOD(..., 'E_t', p=1 or 2, iterative=False), update scalar
  model parameters, and call FE.build_var_form(quasi_static=True). For VTI,
  call assign_native_vti(model, sampler) AFTER build_var_form; do not rebuild
  the upstream variational form after tensor assignment. The sampler supplies
  sample(points, markers, chunk_size) -> (sigma_h, sigma_v), in S/m.
  solve_model then assembles and solves independently of upstream Solver.
  Natural boundaries, one fitted line source, quasi-static/no-IP physics,
  positive scalar inverse permeability and a fixed real current are required.
  Pass receiver_domain when an interface receiver must use one material side.
  The returned dictionary is also model.mor_stats; its receivers rows contain
  index/frequency/data (complex E/H and coordinates). FS.U keeps the last
  emitted field. Operators are released on exit. A numerical failure keeps
  its statistics and raises RuntimeError; solve instead returns failed stats.

mesh_data/
  The block cross-validation and strategy comparisons use different meshes.
  The sphere display scene is separate from the numerical benchmark mesh.
  MR3D H5 files store mesh/domain information, not the full VTI material.
  Keep axes.npz, log_rho_h.npy, log_rho_v.npy and survey inputs with the
  current loader, or develop and validate an equivalent per-cell material
  loader before removing those volumes. Original MR3D data may instead be
  obtained from the cited source when the complete tested preprocessing
  route is included. A source link alone is insufficient for missing derived
  inputs that the current runtime expects directly.
  Document the selected mesh route, companion file names and any final
  checksums here. Generated electromagnetic fields and benchmark results
  belong in outputs/runs/, separately from the mesh/model inputs.

  Mesh identity audit, 2026-11-01:
  - The two block H5 files were directly recounted: 5,256,131 tetrahedra /
    12,252,168 real P1 DOFs for cross-validation, and 309,349 / 720,052 for
    strategy comparisons. Their hashes and generator source match the
    original production archives. Neither tetrahedral generator was rerun.
  - The emg3d 128 x 128 x 80 arrays were regenerated in memory and exactly
    match the frozen NPZ, including material arrays and array hashes.
  - The sphere's three-stage source and production records match 2,519,112
    tetrahedra / 5,905,442 real DOFs. Its formal H5 was not found locally.
    Use the recorded q17 quality=1.7 route and both recorded Plaza stages;
    the old formal_generation_settings.json with quality=1.4 is not the
    final recipe. No from-zero regeneration has been validated here.
  - MR3D's five frozen manifests and measured-run protocol match the paper,
    but the current D-drive generator's adaptive calibration differs from
    the historical M5 sequence. Do not publish its default invocation as a
    verified way to recover those five meshes. A fixed-final-scale route
    is recorded in the audit as a candidate only; it still needs execution
    and count/topology/material checks. Formal M1-M5 H5 files were not found
    locally, and old local M1 files do not have the paper's dimensions.
  Full source hashes, input hashes, evidence paths, production identities,
  exact scales and verification limits are in MESH_AUDIT_20261101.json.

runs/
  Keep fixed case parameters in a clearly labelled block at the beginning
  of each run script: geometry, conductivity/anisotropy, source/receiver
  definitions, frequencies, FE/BC settings, solver options, pole schedules
  and benchmark sweep values. CLI arguments may override documented options.
  No separate config/ directory is planned. Read mesh companion parameters
  from mesh_data/ rather than duplicating them in an independent config.
  Resolve all paths relative to this
  package or explicit CLI arguments, not the original /mnt/d or HPC paths.
  Include the shared helpers actually imported by the release entry points.
  Keep the run_* entry names as planned until they are implemented. Document
  case commands, expected output shapes, numerical acceptance thresholds
  and time/memory estimates in this README. Runtime results belong in
  outputs/runs/. The public workflow needs campaign orchestration only if
  its selected entry points actually depend on it.

figs/
  Bundle only inputs needed by the drawing scripts, which may be CSV, JSON,
  NPZ or VTK scenes. Preserve all linked files of the figure-7 VTM scene.
  Its existing pixel comparison also reads an original PNG; either include
  that validation input or make the comparison optional in the release.
  For figure 11, a small exported section.npz can replace the two original
  SEG-Y volumes for redraws, after extracting/testing the section renderer.
  The section export does not currently exist in this scaffold.
  Figure 12 and tables 4-5 must use the current four-method performance.csv;
  do not substitute historical performance columns from the old scaling.csv.
  Keep published measurements frozen for direct redraws. Rerun results can
  be accepted as alternate inputs when the entry points support them.
  Document units, component ordering, frequencies, metrics and aggregation
  here. Write generated images to outputs/figures/.
  plot_all.py remains proposed, not implemented. No separate table directory
  or table-export script is planned. Literature tables 1-2 stay in the paper;
  numerical tables 3-5 reuse the corresponding case inputs or rerun results.

FILE_FIGURE_MAP.csv
  The mapping includes all 12 main figures. Numerical tables 3-5 retain
  shared-data references only, without separate table files or export
  scripts. Literature tables 1-2 do not require release entries. Every
  planned file is marked as planned rather than already copied. Source
  paths are relative to ../custEM/; release paths are relative to this folder.
  Update the map with tested commands and final output filenames when the
  actual release files have been organized.

CORE REGRESSION COMMANDS
------------------------
Create the environment from the repository root on Linux x86_64 (linux-64).
The YAML pins Conda package versions/builds and installs the official COMET
and custEM repositories at fixed commits through its pip section. The
existing research environment and custEM checkout need no modification:

  conda env create -f environment.yml
  conda activate supermor
  python -m pip install --no-deps --no-build-isolation .
  export PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
  export XDG_CACHE_HOME=/tmp/supermor-core-cache
  unset PYTHONPATH
  python -B -m pytest -q -p no:cacheprovider tests
  mpiexec -n 2 python -B -m pytest -q -p no:cacheprovider tests

Activate the environment, or use conda run, so DOLFIN receives its compiler
and pkg-config paths. Do not use the environment's Python alone by absolute
path. The YAML retains the tested compatible numerical/compiler/rendering
stack; it is not a minimal environment or a cross-platform checksum lock.
Use a different environment name with conda env create -n NAME -f
environment.yml if supermor already exists.

Default tests load core/supermor and reject local/mixed or research-modified
custEM imports. For installed-wheel validation from another working directory:

  SUPERMOR_ROOT="$PWD"
  cd /tmp
  SUPERMOR_TEST_INSTALLED=1 python -B -m pytest -q -p no:cacheprovider "$SUPERMOR_ROOT/tests"
  SUPERMOR_TEST_INSTALLED=1 mpiexec -n 2 python -B -m pytest -q -p no:cacheprovider "$SUPERMOR_ROOT/tests"

The suite tests independent UFL/dense references, actual equation residuals,
P1/P2 sources and E/H sampling, VTI assignment, MPI failure handling and real
MOD/H5 loading. Compilation caches and generated H5 files stay in /tmp.
CORE_VALIDATION.json contains exact counts and completed comparisons.

Independent-process A/B checks can be repeated against an archived old tree
(from the repository root, after returning from /tmp):

  python -B tests/compare_sources.py --legacy-code /path/to/old/core/code --output outputs/core_validation/old1
  PYTHONPATH="$PWD/core" python -B tests/compare_sources.py --output outputs/core_validation/new1
  python -B tests/compare_results.py outputs/core_validation/old1.npz outputs/core_validation/new1.npz

Run both workers with mpiexec -n 2 for the MPI2 A/B pair. Compare new serial
and MPI2 receiver arrays using compare_results.py --cross-mpi. The workers
cover 14 P1/P2 scalar/VTI method combinations at 0.1/1/100 Hz. Fixed-MPI fields
and receivers use relative L2 1e-10 plus absolute 1e-12, with matching algorithm
summaries. Cross-MPI comparisons use receiver E/H only, relative L2 1e-4 plus
absolute 1e-12; partitioned raw DOFs are not compared across MPI sizes.
Independent tiny references require residual <1e-10; emitted solver fields
require residual <=1.01e-6, mass-norm and E/H errors <1e-3. These checks do not
claim a rerun of the large paper benchmarks or their original timing/memory.

PUBLICATION METADATA
--------------------
The existing source license is copied verbatim. THIRD_PARTY.txt records
known input sources; it does not assign new licenses to the authors' data.
State the selected license for newly published datasets before releasing.
A CITATION.cff and a final SHA256SUMS.txt are useful optional additions once
authors, release DOI and the actual files have been confirmed. The current
draft's supplementary section says it is not part of the final paper; if
supplementary figures/tables are published later, extend the mapping then.

NEXT PACKAGING WORK
-------------------
1. Choose final meshes or complete generation workflows for each case.
2. Populate mesh_data/ and runs/ with the required inputs and case entries;
   keep them independent of old absolute paths/protocols. core/ is populated.
3. Populate figs/ with scripts and drawing inputs. Extract the figure-11
   section renderer; update imports/CLI defaults and avoid duplicated data.
4. The independent core environment and small numerical checks are recorded
   here. Validate the remaining full-case and redraw environments, inputs
   and resource requirements before claiming complete paper reproduction.
5. Verify the map, input availability and redraws. Archive the release source
   and dataset with matching version information. Files ignored by Git are
   not automatically included in a source-only release.
