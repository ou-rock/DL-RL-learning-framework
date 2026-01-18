"""Job packaging system for GPU execution"""

import json
import tarfile
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime

from .base import JobConfig


class JobPackager:
    """Packages implementations for remote GPU execution"""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize packager

        Args:
            output_dir: Directory for output bundles (default: temp dir)
        """
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def package(self, job_config: JobConfig) -> Path:
        """Package a job for remote execution

        Args:
            job_config: Job configuration

        Returns:
            Path to the created tarball
        """
        # Create temporary staging directory
        with tempfile.TemporaryDirectory() as staging:
            staging_path = Path(staging)
            project_dir = staging_path / "project"
            project_dir.mkdir()

            # Copy implementation file
            impl_path = Path(job_config.implementation_path)
            if impl_path.exists():
                dest = project_dir / impl_path.name
                dest.write_text(impl_path.read_text())

            # Copy extra files
            for extra_file in job_config.extra_files:
                extra_path = Path(extra_file)
                if extra_path.exists():
                    dest = project_dir / extra_path.name
                    dest.write_text(extra_path.read_text())

            # Generate training script
            train_script = self._generate_train_script(job_config)
            (project_dir / "train.py").write_text(train_script)

            # Generate requirements.txt
            requirements = self._generate_requirements(job_config)
            (project_dir / "requirements.txt").write_text(requirements)

            # Write job configuration
            config_data = {
                "concept": job_config.concept,
                "dataset": job_config.dataset,
                "epochs": job_config.epochs,
                "batch_size": job_config.batch_size,
                "learning_rate": job_config.learning_rate,
                "target_accuracy": job_config.target_accuracy,
                "hyperparameters": job_config.hyperparameters
            }
            (project_dir / "job_config.json").write_text(json.dumps(config_data, indent=2))

            # Create tarball
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bundle_name = f"{job_config.concept}_{timestamp}.tar.gz"
            bundle_path = self.output_dir / bundle_name

            with tarfile.open(bundle_path, "w:gz") as tar:
                tar.add(project_dir, arcname="project")

            return bundle_path

    def _generate_train_script(self, job_config: JobConfig) -> str:
        """Generate training script for the job"""
        impl_name = Path(job_config.implementation_path).stem

        return f'''#!/usr/bin/env python3
"""Auto-generated training script for {job_config.concept}"""

import json
import time
import sys
from pathlib import Path

# Import the implementation
from {impl_name} import *

# Load configuration
with open("job_config.json") as f:
    config = json.load(f)

print(f"Starting training for {{config['concept']}}")
print(f"Dataset: {{config['dataset']}}")
print(f"Epochs: {{config['epochs']}}")
print(f"Target accuracy: {{config['target_accuracy']}}")

# Training loop placeholder
# Real implementation would load dataset and run training
metrics = {{
    "epoch": [],
    "loss": [],
    "accuracy": []
}}

start_time = time.time()

for epoch in range(config['epochs']):
    # Simulated training step
    loss = 1.0 / (epoch + 1)
    accuracy = min(0.99, 0.5 + 0.02 * epoch)

    metrics["epoch"].append(epoch + 1)
    metrics["loss"].append(loss)
    metrics["accuracy"].append(accuracy)

    print(f"Epoch {{epoch + 1}}/{{config['epochs']}}: loss={{loss:.4f}}, accuracy={{accuracy:.4f}}")

    # Check early stopping
    if accuracy >= config['target_accuracy']:
        print(f"Target accuracy reached!")
        break

elapsed = time.time() - start_time

# Save results
results = {{
    "final_accuracy": metrics["accuracy"][-1],
    "final_loss": metrics["loss"][-1],
    "total_epochs": len(metrics["epoch"]),
    "training_time": elapsed,
    "metrics": metrics,
    "passed": metrics["accuracy"][-1] >= config["target_accuracy"]
}}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\\nTraining complete in {{elapsed:.1f}}s")
print(f"Final accuracy: {{results['final_accuracy']:.4f}}")
print(f"Passed: {{results['passed']}}")

sys.exit(0 if results['passed'] else 1)
'''

    def _generate_requirements(self, job_config: JobConfig) -> str:
        """Generate requirements.txt content"""
        reqs = job_config.requirements.copy()

        # Add common requirements if not present
        common = ['numpy', 'torch', 'torchvision']
        for req in common:
            if req not in reqs:
                reqs.append(req)

        return '\n'.join(sorted(reqs))
