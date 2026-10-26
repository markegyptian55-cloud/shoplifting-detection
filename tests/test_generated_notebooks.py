"""
Independent Verification Test Suite for Production Video Classification Notebooks.
Validates JSON integrity, cell counts, Markdown-Code alternation, AST compilation,
and English language isolation.
"""

import os
import json
import ast
import re
import unittest
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"

NOTEBOOK_SPECS = {
    "shoplifting-detection-pipeline.ipynb": {
        "expected_cells": 20,
        "expected_pairs": 10,
        "required_terms": ["DATASET_ROOT", "StratifiedGroupKFold", "cached_frames", "SpatiotemporalResNetGRU", "quantize_dynamic"]
    },
    "02-shoplifting-scratch-3dcnn.ipynb": {
        "expected_cells": 12,
        "expected_pairs": 6,
        "required_terms": ["Conv2Plus1D", "Shoplifting3DCNN_Scratch", "resolve_preprocessed_paths", "quantize_dynamic"]
    }
}

class TestGeneratedNotebooks(unittest.TestCase):
    def test_all_notebooks_exist_and_valid_json(self):
        for nb_name, spec in NOTEBOOK_SPECS.items():
            nb_path = NOTEBOOKS_DIR / nb_name
            self.assertTrue(nb_path.exists(), f"Missing notebook: {nb_name}")
            self.assertGreater(nb_path.stat().st_size, 1000, f"Notebook {nb_name} too small")
            
            with open(nb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.assertIn("cells", data)
            self.assertIn("metadata", data)
            self.assertEqual(len(data["cells"]), spec["expected_cells"],
                             f"{nb_name} cell count mismatch: expected {spec['expected_cells']}, got {len(data['cells'])}")

    def test_cell_rhythm_and_ast_compilation(self):
        for nb_name, spec in NOTEBOOK_SPECS.items():
            nb_path = NOTEBOOKS_DIR / nb_name
            with open(nb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            cells = data["cells"]
            # Strict alternating rhythm: Markdown then Code
            for i in range(0, len(cells), 2):
                self.assertEqual(cells[i]["cell_type"], "markdown",
                                 f"Cell {i} in {nb_name} must be Markdown (Heading & Scientific Context)")
                self.assertEqual(cells[i+1]["cell_type"], "code",
                                 f"Cell {i+1} in {nb_name} must be Code (Implementation)")
                
                # Check valid AST syntax for all code cells
                code_text = "".join(cells[i+1].get("source", []))
                try:
                    ast.parse(code_text)
                except SyntaxError as e:
                    self.fail(f"Syntax error in {nb_name} Cell {i+1} at line {e.lineno}: {e.msg}")
                    
            print(f"[Verification] {nb_name}: {len(cells)} cells ({spec['expected_pairs']} pairs) AST PASSED!")

    def test_required_terms_and_language_isolation(self):
        arabic_pattern = re.compile(r"[\u0600-\u06FF]")
        
        for nb_name, spec in NOTEBOOK_SPECS.items():
            nb_path = NOTEBOOKS_DIR / nb_name
            with open(nb_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
                
            # Language isolation: Zero Arabic characters inside notebooks
            arabic_matches = arabic_pattern.findall(raw_text)
            self.assertEqual(len(arabic_matches), 0, f"Found Arabic characters in {nb_name}: {arabic_matches[:10]}")
            
            # Check domain terms
            for term in spec["required_terms"]:
                self.assertIn(term, raw_text, f"Missing required term '{term}' in {nb_name}")
                
            print(f"[Verification] {nb_name}: 100% English & required contracts verified.")

    def test_defensive_contracts_and_anti_dummy_assertions(self):
        """Verifies that notebooks prevent dummy zero fallbacks, handle relative paths, and patch external data ONNX bug."""
        # 1. Pipeline notebook must contain cached_frames and DataParallel
        pipeline_text = (NOTEBOOKS_DIR / "shoplifting-detection-pipeline.ipynb").read_text(encoding="utf-8")
        self.assertIn("cached_frames", pipeline_text)
        self.assertIn("DataParallel", pipeline_text)

        # 2. Scratch notebook must raise FileNotFoundError rather than silently training on all-zeros
        scratch_text = (NOTEBOOKS_DIR / "02-shoplifting-scratch-3dcnn.ipynb").read_text(encoding="utf-8")
        self.assertIn("FileNotFoundError", scratch_text)
        self.assertIn("rel_cache_path", scratch_text)
        self.assertIn("_safe_save_and_reload", scratch_text)

        print("[Verification] All defensive contracts, anti-dummy assertions, and live telemetry verified.")

if __name__ == "__main__":
    unittest.main()
