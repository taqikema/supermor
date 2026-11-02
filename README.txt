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
the completed local checks and the single formal block dual-pole run. Figure
scripts/data remain planned. VERSION.txt records the formal numerical
versions; detailed migration-source audits are archived outside this checkout.
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

RELEASE TREE (CORE, MESH AND RUNS POPULATED; FIGURES PLANNED)
-----------------------------------------------------------
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
    |-- case1_block/
    |-- case2_layered_sphere/
    |-- case3_mr3d/
    `-- figures/

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
  migration. Generated EM fields and benchmark results belong in outputs/runs/.
  The supplied MR3D fetch/prepare commands with --model original can create
  the material arrays from the public model; generate.py all does not run
  this preprocessing. The current prepare command expects raw survey and
  original interface files as well as the two SEG-Y files. fetch downloads
  that complete raw input set. Future runs must check material inputs too.

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
  drawing/run file is marked as planned rather than already copied. Mesh
  paths now refer to populated data/mesh_data/ directories, while the notes
  distinguish their migrated generators from the pending plotting entries.
  Source paths are relative to ../custEM/; release paths are relative to this folder.
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
1. Package the selected bulk mesh/model data for the separate dataset archive;
   retain the tested mesh_gen workflows and record the real download/DOI later.
2. Implement the planned runs/ case entries using runs/common/mesh_inputs.py;
   complete MR3D forward material inputs/preprocessing and validate the actual
   numerical case paths. The core and mesh contents are already populated.
3. Populate figs/ with scripts and drawing inputs. Extract the figure-11
   section renderer; update imports/CLI defaults and avoid duplicated data.
4. The independent core environment and small numerical checks are recorded
   here. Validate the remaining full-case and redraw environments, inputs
   and resource requirements before claiming complete paper reproduction.
5. Verify the map, input availability and redraws. Archive the release source
   and dataset with matching version information. Files ignored by Git are
   not automatically included in a source-only release.

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
