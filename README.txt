SUPERMOR paper reproduction package
==================================

STATUS
------
The independent supermor package is in core/supermor; numerical tests are
in tests/. It extends a separately installed, unmodified official custEM.
The numerical environment and pinned upstream sources are recorded in
root environment.yml; installation commands are provided below.
CORE_VALIDATION.json records the completed migration checks and their limits.
Mesh generators are implemented in mesh_gen/. Their necessary inputs and
previously validated local outputs are organized under data/mesh_data/.
MESH_VALIDATION.json records the mesh migration checks and verification limits.
Numerical run entries are implemented in runs/. RUNS_VALIDATION.json records
the completed local checks and the single formal block dual-pole run. Nine
figure entries redraw all twelve main figures. Their completed redraw checks
are summarized below; detailed migration audits are archived outside this
checkout. VERSION.txt records the numerical source versions.
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

RELEASE TREE
------------
supermor/
|-- README.txt
|-- pyproject.toml                   SUPERMOR package/version and upstream pin
|-- requirements.txt                 Plotting/data dependencies
|-- environment.yml                 Version/build-pinned linux-64 environment
|-- CORE_VALIDATION.json             Migration results and acceptance limits
|-- ENVIRONMENT.txt                  Numerical stack, exact versions and commands
|-- LICENSE.rst                      Existing custEM LGPLv3/GPL license text
|-- THIRD_PARTY.txt                  Input sources and existing attribution
|-- VERSION.txt                      Source commit; add final release tag/DOI
|-- FILE_FIGURE_MAP.csv               Paper item -> source -> tested release path
|-- MESH_AUDIT_20261101.json          Immutable earlier provenance audit snapshot
|-- MESH_VALIDATION.json             Current mesh migration/results and limits
|-- RUNS_VALIDATION.json             Local run and regression evidence
|-- core/
|   `-- supermor/
|       |-- __init__.py              Independent public API
|       |-- methods.py               Operator/model solver interfaces
|       |-- full_order.py            Production MPI DFOM/IFOM
|       |-- mor.py                   RK/SUPERMOR numerical kernels
|       |-- operators.py             P1/P2 assembly, line source, E/H sampling
|       |-- materials.py             Native distributed DG0 VTI assignment
|       |-- mor_high_order.py        P2 exact-sequence/AMS transfers
|       `-- mor_mpi_legacy.py        P1 historical sphere-MPI Kg MR variant
|-- tests/                           Algebra, P1/P2, official H5 and A/B checks
|-- mesh_gen/
|   |-- generate.py                  Unified mesh generation/check entry
|   |-- paths.py                     Shared case -> runtime input paths
|   |-- verify.py                    Independent topology/edge checks
|   |-- case1_block/
|   |   |-- cross_validation/        mesh_custem.py, refine_receiver_mesh.py,
|   |   |                           validation_common.py, emg3d_worker.py
|   |   `-- strategy/                mesh_strategy.py, strategy_contract.py
|   |-- case2_layered_sphere/        Initial generator, two Plaza entries,
|   |                               and refine_terrain_rx_star.py helper
|   `-- case3_mr3d/
|       `-- experiments/mr3d_custem/ mr3d.py, mesh.py, prepare.py
|-- data/
|   |-- figure_data/                 Small model JSON and complete figure-7 VTK scene
|   |-- raw_data/case3_mr3d/         Two complete SEG-Y inputs (local/downloaded)
|   |-- model_data/                  Prepared physical model arrays (local/Zenodo)
|   |-- reference_data/              Frozen numerical validation responses
|   `-- mesh_data/                   Local/Zenodo input and mesh artifacts
|       |-- case1_block/
|       |   |-- cross_validation/
|       |   |   |-- custem/
|       |   |   |   |-- parent/meshes/custem_L2/
|       |   |   |   `-- final/meshes/custem_L2/
|       |   |   `-- emg3d/meshes/emg3d_fixed/
|       |   `-- strategy/mesh/       Separate gradient/pole mesh
|       |-- case2_layered_sphere/
|       |   |-- inputs/sphere_surface_template.npz
|       |   |-- candidate_q17_local025/attempt_01/
|       |   |-- q17_rx_plaza_round_01/
|       |   `-- q17_rx_plaza_earth_round_02/
|       `-- case3_mr3d/
|           |-- prepared/           Original-model metadata, survey, six interfaces
|           `-- meshes/             M1/ through M5/, each with attempt_01/
|-- runs/
|   |-- common/                     Mesh paths, PSS, result IO, receiver metrics
|   |-- case1_block/
|   |   |-- cross_validation/        run_cross_validation.py and four workers/helpers
|   |   `-- strategy/                run_strategy.py and worker.py
|   |-- case2_layered_sphere/
|   |   |-- run_methods_mpi.py
|   |   |-- run_mpi_scaling.py
|   |   `-- run_frequency_count.py
|   `-- case3_mr3d/
|       |-- prepare_materials.py
|       |-- mr3d_native.py
|       |-- run_methods_mpi.py
|       `-- run_mesh_scaling.py
|-- figs/
|   |-- common/                     Styles, VTK renderer and result adapters
|   |-- case1_block/                Four scripts: figures 1-6
|   |-- case2_layered_sphere/        Three scripts: figures 7-10
|   `-- case3_mr3d/                 Two scripts: figures 11-12
`-- outputs/                         Local files excluded from Git
    |-- case1_block/{cross_validation,strategy}/
    |-- case2_layered_sphere/{mpi_scaling,frequency_count}/
    `-- case3_mr3d/mesh_scaling/
        Each experiment retains paper/ inputs and separate new-run directories.

RELEASE CONTENT NOTES
---------------------
core/
  Seven shared Python modules provide the paper extension. A separate
  mor_mpi_legacy.py preserves the three historical P1 Kg MR definitions for
  the sphere MPI experiment. No custEM source tree,
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

mesh_gen/ and data/mesh_data/
  mesh_gen/ contains the actual mesh-generation source. All required case
  inputs and generated mesh files belong under data/mesh_data/. No mesh H5,
  XML, VTK, TetGen intermediates, package installations or caches belong in
  the mesh_gen/ source directories. Block cross-validation and strategy
  comparisons use different meshes. The sphere figure-7 display scene is
  separate from the numerical benchmark mesh.

  Use the activated numerical environment. From this repository root:

    python -B mesh_gen/generate.py --list
    python -B mesh_gen/generate.py all --dry-run
    python -B mesh_gen/generate.py all --check
    python -B mesh_gen/generate.py all

  The small path/topology regression tests are in tests/test_mesh_generation.py:

    python -B -m pytest -q -p no:cacheprovider tests/test_mesh_generation.py

  Available choices are block_cross, block_emg3d, block_strategy, sphere,
  mr3d, mr3d_M1 through mr3d_M5, and all. The unified entry supplies the
  recorded formal parameters; individual historical entry defaults are not
  a substitute for that formal recipe. Existing validated outputs are reused
  without overwriting them. --check independently recounts tetrahedral H5
  topology/unique edges and checks the emg3d arrays; it is not a forward run.
  The emg3d check also requires and validates receiver_support.npz/json.
  Its generation entry restores either missing sampling sidecar from the
  fixed mesh and receiver definitions, while rejecting inconsistent retained
  data. No separate manually prepared sampling file is needed.
  --dry-run shows commands without starting the large mesh generation.

  To generate in a separate new data tree, pass an explicit data root:

    python -B mesh_gen/generate.py block_strategy --data-root /tmp/supermor-new-mesh-data
    python -B mesh_gen/generate.py sphere --data-root /tmp/supermor-new-mesh-data

  The alternate root replaces data/mesh_data, not the repository root. The
  entry copies missing canonical sphere/MR3D inputs from the default local
  data root when available. Sphere needs its inputs/sphere_surface_template.npz;
  MR3D needs prepared/manifest.json, survey.npz and six interfaces/*.xyz.
  Download/provide those interfaces first when they are absent locally.
  Data-root overrides and relative package paths replace old D-drive/HPC
  runtime paths. Old absolute paths retained in copied refinement records
  are historical provenance only. Migration manifests disclose any rewrites
  and source adaptations instead of claiming original metadata byte identity.

  Runtime entries should use the same path registry:

    from runs.common.mesh_inputs import get_mesh
    files = get_mesh('sphere', require=True)
    mesh_h5 = files['mesh']
    parameters_json = files['parameters']

  get_mesh accepts block_cross, block_emg3d, block_strategy, sphere, and
  mr3d_M1 through mr3d_M5. It returns absolute Path values for the required
  mesh/array files and companion inputs. data_root=PATH selects an alternate
  tree. The final sphere runtime directory is
  data/mesh_data/case2_layered_sphere/q17_rx_plaza_earth_round_02/meshes/.
  Local MR3D regeneration uses attempt_01 names at each fixed final scale;
  those names do not replace the recorded historical M3/M5 provenance.

  Completed local regeneration and verification, 2026-11-01:
  - Block cross-validation: 5,256,131 tetrahedra, 6,126,084 unique edges,
    12,252,168 real P1 DOFs. Strategy mesh: 309,349 tetrahedra, 360,026
    unique edges, 720,052 real P1 DOFs. Both were generated from scratch;
    vertex coordinates, connectivity, per-cell regions and complete source/
    receiver coordinates exactly match the locally available formal meshes.
  - emg3d: regenerated 128 x 128 x 80 axes/material arrays exactly match
    the frozen NPZ arrays.
  - Sphere: quality=1.7 initial mesh plus both recorded Plaza rounds were
    generated from scratch. Final counts are 2,519,112 tetrahedra, 2,952,721
    unique edges and 5,905,442 real P1 DOFs. Geometry, regions and all source/
    receiver checks pass. Its original formal H5 is still unavailable for
    direct coordinate/connectivity comparison; count agreement and the
    recorded checks do not establish that stronger identity claim.
  - MR3D: all five final fixed scales were executed once; independently
    recounted tetrahedra/edges/real P1 DOFs match the paper for M1-M5.
    M5 completed meshing, then its conversion was interrupted by SIGTERM;
    standalone unchanged vtk2xml2h5 converted the retained VTK successfully.
    No second M5 meshing occurred. Recovery provenance and unknown original
    Python elapsed time are recorded. The original formal H5 arrays have
    not been compared, and no full MR3D forward benchmark was rerun here.
  These validated outputs were copied into this data tree. The source-path
  migration is recorded separately from the original regeneration. Current
  evidence is in MESH_VALIDATION.json. MESH_AUDIT_20261101.json is an immutable
  earlier snapshot whose then-unverified statuses are historical, not the
  current mesh-generation status.

  Git tracks the small sphere template and MR3D prepared/manifest.json plus
  survey.npz. Bulk H5 files, six MR3D interface XYZ files and other generated
  artifacts are local data excluded from Git. They must be included in the
  separate dataset archive if selected for distribution; a Git source-only
  clone does not contain them. The dataset DOI/download address has not yet
  been assigned. Do not treat placeholder archive metadata as a live source.

  File-generation contract:
  - The sphere template, MR3D prepared/manifest.json, survey.npz and the six
    interface XYZ files are independent inputs supplied with Git/the dataset.
  - Mesh H5 files, companion parameters, mesh manifests, sphere source/receiver
    CSV and geometry/source audits, and emg3d receiver sampling files are
    generated by the case workflows. Intermediate meshes and refinement
    mappings are generated too; migration_manifest.json is packaging evidence
    and is not read by the numerical model.

  MR3D mesh-generation inputs are complete. The H5 files contain mesh/domain
  information, not the full VTI material required by future forward runs.
  Those runs still need axes.npz, log_rho_h.npy, log_rho_v.npy and survey data,
  or a complete tested preprocessing route that regenerates them. These
  material volumes and the numerical run entries are not part of this mesh
  migration. Generated EM fields and benchmark results belong under outputs.
  runs/case3_mr3d/prepare_materials.py prepares material arrays from the two
  original SEG-Y volumes alone; generate.py all does not run this stage.
  The older mesh-preparation workflow also consumes survey/interfaces for
  geometry, separately from this material-only route.

runs/
  Keep fixed case parameters in a clearly labelled block at the beginning
  of each run script: geometry, conductivity/anisotropy, source/receiver
  definitions, frequencies, FE/BC settings, solver options, pole schedules
  and benchmark sweep values. CLI arguments may override documented options.
  No separate config/ directory is planned. Read mesh companion parameters
  from data/mesh_data/ through the shared helper rather than duplicating
  them in an independent config.
  Resolve all paths relative to this
  package or explicit CLI arguments, not the original /mnt/d or HPC paths.
  Include the shared helpers actually imported by the release entry points.
  Case commands and the completed numerical checks are documented below.
  Full-case resource estimates remain separate from drawing requirements.
  Runtime results belong in outputs/<case>/<experiment>/.

figs/
  Nine standalone entries redraw figures 1-12. Default inputs for result
  plots are the frozen paper/ data under outputs/<case>/<experiment>/.
  No latest-run selection or implicit use of a previous research checkout
  occurs. --run-dir PATH explicitly selects new numerical results; adapters
  check the required configurations and preserve their own measurement
  provenance. Derived drawing inputs may be written under that run's
  plot_inputs/, leaving its raw results and the paper/ inputs unchanged.
  --input-dir PATH selects another frozen drawing-input directory. These
  two options are mutually exclusive. --output-dir PATH overrides the
  default output directory, which is the drawing script's own directory.
  Each image and metadata file has a unique figNN_ prefix.

  Figure 1 reads data/figure_data/case1_block/geometry/model_parameters.json.
  Figure 7 reads data/figure_data/case2_layered_sphere/geometry/survey_scene.vtm
  and all seven referenced VTU/VTP files. Its display scene is independent
  of the large numerical mesh. No original PNG is a runtime input.
  Figure 11 reads the complete original Horizontal_resistivity.sgy and
  Vertical_Resistivity.sgy under data/raw_data/case3_mr3d/, plus the survey
  already supplied in data/mesh_data/case3_mr3d/prepared/survey.npz. It reads
  only the two native y lines bracketing y=-253 m, interpolates log10(rho),
  and keeps the original coordinate/depth conventions. No section NPZ or
  historical scaling summary is needed. These raw volumes can also feed
  runs/case3_mr3d/prepare_materials.py; no second model copy is required.

  Result-input directories:
    outputs/case1_block/cross_validation/paper/             Figures 2-4
    outputs/case1_block/strategy/paper/gradient/            Figure 5
    outputs/case1_block/strategy/paper/poles/               Figure 6
    outputs/case2_layered_sphere/mpi_scaling/paper/         Figure 8
    outputs/case2_layered_sphere/frequency_count/paper/     Figures 9-10
    outputs/case3_mr3d/mesh_scaling/paper/                  Figure 12

  Git tracks the figure code and figure-1 model JSON. The complete figure-7
  VTK scene, frozen result data, full SEG-Y volumes, rendered images and
  rendering metadata are present locally but ignored by Git. A source-only
  clone needs the figure-7 scene and frozen result inputs from the separate
  dataset, plus the original model volumes. No dataset DOI has been assigned
  yet. FILE_FIGURE_MAP.csv lists the current inputs and redraw commands;
  completed verification and its limits are summarized below.

  Metric conventions:
  - Figures 3-4 use per-component complex symmetric NRMSD in percent:
    200*abs(a-b)/(abs(a)+abs(b)), four frequencies and 3321 receivers.
  - Figures 5-6 use maximum pointwise three-complex-component vector
    relative error over receivers, not the whole-array relative L2.
  - Time is complete-forward seconds; memory is sampled aggregate job PSS
    in GiB (2**30 bytes). Speedups retain the documented baseline.
  - Figure 12 retains 20 configuration statuses, 17 measured configurations
    and empty DFOM M3-M5 entries in the frozen paper data. New runs retain
    actual repeats and status. Missing required curves are errors.
  No separate tables directory or exporter is supplied. Tables 3-5 reuse
  corresponding drawing inputs; literature tables 1-2 remain in the paper.

FILE_FIGURE_MAP.csv
  The map includes all twelve main figures and shared-data references for
  numerical tables 3-5. Source paths refer to ../custEM/; migrated paths are
  relative to this checkout. Commands and unique generated filenames are
  recorded alongside each entry. Source paths are provenance only and are
  not used to discover missing runtime inputs.

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
1. Select and archive the ignored mesh/model and frozen drawing inputs;
   record the actual dataset address/DOI and license before publication.
2. Generate or supply the full MR3D VTI material arrays before its forward
   runs; the raw SEG-Y inputs and tested preparation entry are available.
3. Validate the remaining full numerical campaigns and resource needs.
   Redraw validation does not establish a rerun of those large cases.
4. Assign the source release tag and matching archive metadata. Ignored
   files do not appear automatically in a Git-only release.

NUMERICAL RUNS
--------------
Inputs resolve relative to this checkout, independent of the working directory.
For runs, --data-root replaces the complete data/ directory; the mesh generator's
--data-root instead replaces data/mesh_data/. get_case_mesh() makes this
separation explicit. Generated H5 and companion parameters are consumed together.
Other model inputs live under data/model_data/. Existing reference responses
can be supplied under data/reference_data/; they are optional when computing
new reference responses locally. New results, native custEM files, logs and
sampled aggregate job PSS belong under outputs/. Each coordinator creates a
new run directory and rejects overwriting completed results. Input data is
read through a file-symlink view in outputs/runtime_mesh because upstream
custEM otherwise creates auxiliary directories beside its input mesh.

Activate the numerical environment and use one thread per MPI rank. All
entry scripts load this checkout's core/supermor, not an older installed
supermor package. custEM remains an external model/mesh dependency. The
recorded local block test reuses the existing WSL cust_env without installing
or modifying packages. Its custEM import resolves to the external research
checkout; the independent package's official-dependency regression checks
use the already existing reconstructed environment, recorded separately.
No environment or installed package is copied into this repository.

Block strategy (single dual-pole configuration, 20 frequencies):

  python -B runs/case1_block/strategy/run_strategy.py --strategy dual_pole --save-fields --compute-reference

The separate independent UFL/MUMPS reference is computed only when requested.
If a frozen same-input reference is available, use --reference PATH/receivers.npz
instead of --compute-reference. --strategy all launches the five comparison
configurations; --strategy single_pole selects the single-pole comparison.
--dry-run and worker.py --check-inputs inspect settings/files without a solve.
Published max-pointwise E/H errors use each receiver's three complex components,
then the maximum across receivers; whole-array L2 is a separate saved metric.

Other HPC entries (use --help for method, mesh and launcher overrides):

  python -B runs/case1_block/cross_validation/run_cross_validation.py --dry-run
  python -B runs/case2_layered_sphere/run_mpi_scaling.py --dry-run
  python -B runs/case2_layered_sphere/run_frequency_count.py --dry-run
  python -B runs/case3_mr3d/run_mesh_scaling.py --dry-run

MPI sphere SUPERMOR defaults to --kernel paper_mpi: the historical Kg residual
QR/MR variant from commit 86c71ea. Frequency-count SUPERMOR uses current core,
matching its no-Kg method definition. Only P1 is supported by the historical
variant. Current DFOM/IFOM/RK share the existing production implementations.
The old DFOM1 measurement remains historical provenance, not a performance
value that a new execution can guarantee. No full sphere scan was rerun here.

MR3D needs model_data/case3_mr3d/manifest.json, axes.npz, log_rho_h.npy and
log_rho_v.npy in addition to its generated meshes and survey. These material
arrays are not yet present in this checkout. They are prepared from the two
original SEG-Y volumes, independently of mesh generation:

  python -B runs/case3_mr3d/prepare_materials.py --check
  python -B runs/case3_mr3d/prepare_materials.py --raw-dir /path/to/original-SGYs

--fetch explicitly downloads the two original volumes when desired. Geometry,
survey and the six interface inputs are retained separately in mesh_data.
No MR3D material preparation or full forward calculation was run during this
migration. Its default scan is the paper's 17 configurations / 37 runs,
with three SUPERMOR/IFOM repeats and single RK/DFOM measurements. DFOM M3-M5
require an explicit opt-in. The three historical measurement batches are
recorded, while the portable monitor identifies itself as a new implementation.
Timing uses the worker's complete-forward maximum-rank wall time; memory uses
sampled aggregate job PSS excluding the monitor. Rerun performance is separate
from the published values. The final clean RK M5 and DFOM M2 historical raw
measurement files are still unavailable locally.

No nested README or per-case config directory is required. Formal parameters
are fixed explicitly in the run entries. Receiver arrays, residuals and runtime
identities are saved with each result. Drawing scripts remain a separate step.

Run entry consolidation, 2026-11-02
---------------------------------
MR3D imports native VTI assignment directly from core/supermor/materials.py.
The block DFOM worker now applies its formal MUMPS controls itself; no separate
DFOM launch wrapper is needed. The three controls retain their original prefix
and values: hed_dfom_sym2_mat_mumps_icntl_4=2, icntl_10=0, icntl_22=0.
The frequency-count and MPI-scaling entries remain separate user commands.
The migration-source audit was moved outside the checkout; runtime scripts do
not depend on an audit JSON. VERSION.txt retains the formal source versions,
and RUNS_VALIDATION.json separates the earlier numerical run from later code
consolidation checks. The consolidation does not rerun the large paper cases.

PAPER FIGURE COMMANDS
---------------------
Reuse the existing WSL drawing environment; no package is installed inside
this checkout. All commands below are run from the repository root:

  conda activate cust_env
  export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
  export XDG_CACHE_HOME=/tmp/supermor-figure-cache
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  python -B figs/case1_block/plot_sumor_block_mesh_section.py
  python -B figs/case1_block/plot_sumor_block_cross_validation.py
  python -B figs/case1_block/plot_sumor_hpc_formulation_costs_errors.py
  python -B figs/case1_block/plot_sumor_hpc_pole_costs_errors.py
  python -B figs/case2_layered_sphere/plot_sumor_layered_sphere_survey.py
  python -B figs/case2_layered_sphere/plot_supermor_paper_mpi_performance.py
  python -B figs/case2_layered_sphere/plot_sumor_frequency_count_four_methods.py
  python -B figs/case3_mr3d/plot_mr3d_section.py
  python -B figs/case3_mr3d/plot_sumor_mr3d_scaling_four_methods.py

PNG is supplied for all figures; figures 2-4 also supply PDF, figure 7 uses
PNG only, and the remaining entries supply PNG/PDF/SVG. Times New Roman is
required except for figures 2-4, which retain their original DejaVu style.
Set SUMOR_PAPER_FONT_DIR if the font is not available at a detected system
location. No font files are copied into this repository. VTK uses offscreen
rendering and needs a working EGL/OpenGL implementation.

Example with new numerical results and an explicit image directory:

  python -B figs/case3_mr3d/plot_sumor_mr3d_scaling_four_methods.py --run-dir outputs/case3_mr3d/mesh_scaling/RUN_NAME --output-dir /tmp/mr3d-new-figures

The focused drawing tests do not use the numerical suite's custEM provenance
fixture. In cust_env, run them without that unrelated numerical conftest:

  PYTHONPATH="$PWD" python -B -m pytest --noconftest -q -p no:cacheprovider tests/test_plot_block_inputs.py tests/test_plot_sphere_inputs.py tests/test_plot_mr3d_inputs.py

These tests check result adapters using small arrays and measurement records;
they start no forward solve or MPI campaign. Existing numerical tests retain
their official-custEM guard and their separate environment requirements.

Figure validation, 2026-11-02
-----------------------------
All twelve decoded PNG images are pixel-identical to fresh outputs of the
original entries in the same cust_env. Figure 11's x/z coordinates, both
1022-by-1229 log-resistivity sections and all plotted survey coordinates
are exactly equal. The combined focused drawing suite passes 37 tests.
All nine entries also pass actual default rendering from the repository
root, their own script directory and an unrelated /tmp working directory.
The 31 image files pass format checks; explicit output overrides and all
new-run adapters were exercised. Input hashes and Git inclusion/exclusion
checks pass. No full numerical campaign was executed in this figure work.
Detailed migration evidence and the internal assembly checklist are archived
outside this source release. Neither is required by the drawing scripts.
