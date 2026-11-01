SUPERMOR paper reproduction package: directory plan
==================================================

STATUS
------
This folder currently contains a directory scaffold, a source-to-paper
mapping, candidate dependencies, environment notes and the unchanged
custEM license. It is not yet an executable or complete release. Numerical
inputs, solver source and plotting files have not been relocated here.
Names of scripts and data files in the tree below are proposed release
locations; only the scaffold/documentation files currently exist.
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

PROPOSED COMPLETE RELEASE TREE
------------------------------
supermor/
|-- README.txt
|-- requirements.txt                 Candidate Python packages; lock after testing
|-- ENVIRONMENT.txt                  Numerical stack, exact versions and commands
|-- LICENSE.rst                      Existing custEM LGPLv3/GPL license text
|-- THIRD_PARTY.txt                  Input sources and existing attribution
|-- VERSION.txt                      Source commit; add final release tag/DOI
|-- FILE_FIGURE_MAP.csv               Paper item -> source -> proposed release path
|-- MESH_AUDIT_20261101.json          Local source/artifact/parameter audit evidence
|-- core/
|   |-- setup.py                     Preserve required installation files
|   `-- custEM/                      Installable package and required modules/data
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
  Include the installable custEM package used by the paper and necessary
  installation resources. Copying only custEM/core/mor.py is insufficient:
  the solver also calls FEM assembly, mesh/model loaders and package helpers.
  Preserve module structure and the upstream copyright/license notices.
  The existing source setup.py reads README.rst and custEM/VERSION.rst.
  When packaging it here, adapt its description-file path to this root
  README.txt and preserve the required version resource. No separate
  core/README is needed. Record the packaged source version in VERSION.txt.

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
2. Populate core/, mesh_data/ and runs/ with the required source/inputs and
   make the numerical loaders independent of old absolute paths/protocols.
3. Populate figs/ with scripts and drawing inputs. Extract the figure-11
   section renderer; update imports/CLI defaults and avoid duplicated data.
4. Freeze and validate the real environment, then record the working commands
   in this English root README. Include a small numerical check.
5. Verify the map, input availability and redraws. Archive the release source
   and dataset with matching version information. Files ignored by Git are
   not automatically included in a source-only release.
