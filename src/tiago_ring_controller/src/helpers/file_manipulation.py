import os
import re


class FileManager:
    """Manage file and directory operations."""

    def get_next_results_dir(self, base_name: str, script_file: str, parent_dir: str = "results") -> str:
        """Create next numbered results directory at project root."""
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(script_file)))),
            parent_dir,
            base_name,
        )
        os.makedirs(results_dir, exist_ok=True)

        results_index = 1
        while os.path.exists(os.path.join(results_dir, f"results_{results_index}")):
            results_index += 1

        output_dir = os.path.join(results_dir, f"results_{results_index}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_analysis_results_dir(self, results_file: str, base_name: str) -> str:
        """Create analysis results directory matching input results file number.
        
        Extracts results_N from input file and creates results/base_name/results_N/ at project root.
        """
        match = re.search(r"results_(\d+)", results_file)
        if not match:
            raise ValueError(f"Could not extract results number from: {results_file}")
        
        results_num = match.group(1)
        # Go up 5 levels: results_1 -> single_joint_ring_component -> train -> outputs -> tiago_ring_controller (project root)
        analysis_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(results_file)))))),
            "results",
            base_name,
            f"results_{results_num}",
        )
        os.makedirs(analysis_dir, exist_ok=True)
        return analysis_dir

