#!/usr/bin/env python3
"""Render the saved survey for the paper, omitting outer description blocks.

Run in the activated cust_env environment. No mesh or survey is regenerated.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import vtk

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figs.common import sumor_survey_render
from figs.common.sumor_survey_render import render_survey
from figs.common.sumor_paper_figure_style import TEXT_PT, configure
from figs.common.paths import ROOT, fingerprint

SCENE = ROOT / "data/figure_data/case2_layered_sphere/geometry/survey_scene.vtm"
OUTPUT = Path(__file__).resolve().parent
NAME = "fig07_layered_sphere"


def main(scene_path, output):
    referenced = [scene_path.parent / node.attrib["file"]
                  for node in ET.parse(scene_path).iter("DataSet")]
    scripts = [Path(__file__).resolve(), Path(sumor_survey_render.__file__).resolve()]
    inputs = [scene_path, *referenced]
    before = [fingerprint(path) for path in inputs]
    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(str(scene_path))
    reader.Update()
    scene = reader.GetOutput()
    assert scene.GetNumberOfBlocks() == 7
    blocks = [scene.GetBlock(i) for i in range(7)]
    assert [block.GetNumberOfPoints() for block in blocks[2:]] == [6, 200, 200, 2, 1]
    output.parent.mkdir(parents=True, exist_ok=True)
    render_survey(*blocks, output, show_descriptions=False)
    # VTK sizes text in pixels. Declare the physical width used to convert
    # uniform-point labels to those pixels, so Word does not infer 96 dpi.
    from PIL import Image
    with Image.open(output) as rendered:
        pixels = rendered.copy()
    dpi = pixels.width / (15.92 / 2.54)
    pixels.save(output, dpi=(dpi, dpi))
    assert before == [fingerprint(path) for path in inputs], "Input scene changed during rendering"

    from matplotlib.font_manager import FontProperties, findfont
    configure()
    font_file = findfont(FontProperties(family='Times New Roman'), fallback_to_default=False)
    metadata = {
        "case": 2, "paper_figure": 7,
        "purpose": "Original survey scene rendered without outer descriptions",
        "formats": ["png"], "vector_version_available": False,
        "copied_without_changes": False, "original_inputs_unchanged": True,
        "mesh_or_survey_regenerated": False, "inputs": before,
        "script": fingerprint(__file__), "render_scripts": [fingerprint(path) for path in scripts],
        "outputs": {"png": fingerprint(output)},
        "removed_annotations": ["Top-left survey description", "Bottom-left model description"],
        "retained_annotations": ["Tx: 500 m", "2 km (horizontal)",
                                 "Rx: 10 lines x 20 points", "50 m point / line spacing"],
        "typography": {"font_family": "Times New Roman", "font_file": font_file,
                       "final_width_cm": 15.92, "annotation_pt": TEXT_PT,
                       "vtk_text_size_pixels": round(TEXT_PT * 2100 / (15.92 / 2.54) / 72),
                       "actual_annotation_pt": round(TEXT_PT * 2100 / (15.92 / 2.54) / 72)
                                               * 72 * (15.92 / 2.54) / 2100,
                       "render_width_pixels": 2100},
        "scene_blocks": [{"index": i, "points": block.GetNumberOfPoints(),
                          "cells": block.GetNumberOfCells()} for i, block in enumerate(blocks)],
    }
    destination = output.with_name(output.stem + "_metadata.json")
    destination.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "scene_inputs_unchanged": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SCENE.parent,
                        help="Directory containing survey_scene.vtm and its relative blocks")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    main((args.input_dir / "survey_scene.vtm").resolve(),
         (args.output_dir / (NAME + ".png")).resolve())
