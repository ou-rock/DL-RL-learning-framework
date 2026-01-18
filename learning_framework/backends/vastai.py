"""Vast.ai GPU backend implementation"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .base import GPUBackend, JobConfig, JobStatus, JobResult
from .packager import JobPackager
from .cost_controller import CostController


class VastaiBackend(GPUBackend):
    """Vast.ai GPU backend for remote job execution

    Handles:
    - Finding and renting GPU instances
    - Uploading job bundles via SFTP
    - Running training in tmux sessions
    - Monitoring progress via SSH
    - Downloading results and cleanup
    """

    API_BASE = "https://console.vast.ai/api/v0"

    def __init__(self, config: Dict[str, Any]):
        """Initialize Vast.ai backend

        Args:
            config: Configuration including:
                - api_key: Vast.ai API key
                - daily_budget: Maximum daily spending
                - max_job_cost: Maximum cost per job
                - preferred_gpu: Preferred GPU model
                - min_gpu_ram: Minimum GPU RAM in GB
        """
        super().__init__(config)

        self.api_key = config.get('api_key', '')
        self.preferred_gpu = config.get('preferred_gpu', 'RTX 3060')
        self.min_gpu_ram = config.get('min_gpu_ram', 8)

        # Initialize components
        self.packager = JobPackager()
        self.cost_controller = CostController(
            daily_budget=self.daily_budget,
            max_job_cost=self.max_job_cost
        )

        # Track active jobs
        self._active_jobs: Dict[str, Dict[str, Any]] = {}

    def _api_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make API request to Vast.ai

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request data

        Returns:
            API response as dict
        """
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        url = f"{self.API_BASE}/{endpoint}"

        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, params=data)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=data)
        elif method.upper() == 'PUT':
            response = requests.put(url, headers=headers, json=data)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unknown method: {method}")

        response.raise_for_status()
        return response.json()

    def list_available_gpus(self) -> List[Dict[str, Any]]:
        """List available GPU instances from Vast.ai"""
        try:
            # Query offers with filters
            params = {
                'verified': 'true',
                'disk_space': {'gte': 20},
                'gpu_ram': {'gte': self.min_gpu_ram * 1024}  # Convert to MB
            }

            response = self._api_request('GET', 'bundles', params)
            offers = response.get('offers', [])

            # Filter and format offers
            return [
                {
                    'id': offer['id'],
                    'gpu_name': offer.get('gpu_name', 'Unknown'),
                    'gpu_ram': offer.get('gpu_ram', 0) / 1024,  # Convert to GB
                    'hourly_cost': offer.get('dph_total', 0),
                    'reliability': offer.get('reliability2', 0),
                    'location': offer.get('geolocation', 'Unknown')
                }
                for offer in offers
            ]
        except Exception as e:
            # Return empty list if API fails
            return []

    def estimate_cost(self, job_config: JobConfig) -> float:
        """Estimate cost for running a job"""
        # Get current GPU pricing
        gpus = self.list_available_gpus()

        if not gpus:
            # Use default estimate if no offers available
            hourly_rate = 0.20  # Default estimate
        else:
            # Find cheapest suitable GPU
            suitable = [g for g in gpus if g['gpu_ram'] >= self.min_gpu_ram]
            if suitable:
                hourly_rate = min(g['hourly_cost'] for g in suitable)
            else:
                hourly_rate = gpus[0]['hourly_cost']

        # Estimate runtime based on epochs and dataset
        estimated_hours = job_config.max_runtime_minutes / 60

        return round(hourly_rate * estimated_hours, 2)

    def submit_job(self, job_config: JobConfig) -> str:
        """Submit job to Vast.ai"""
        # Estimate cost and check budget
        estimated_cost = self.estimate_cost(job_config)

        if not self.cost_controller.can_spend(estimated_cost):
            raise ValueError(
                f"Job cost ${estimated_cost:.2f} exceeds budget. "
                f"Remaining: ${self.cost_controller.get_remaining_budget():.2f}"
            )

        # Package job
        bundle_path = self.packager.package(job_config)

        # Find best GPU offer
        gpus = self.list_available_gpus()
        if not gpus:
            raise ConnectionError("No GPU instances available")

        # Select cheapest suitable GPU
        suitable = [g for g in gpus if g['gpu_ram'] >= self.min_gpu_ram]
        if not suitable:
            suitable = gpus

        offer = min(suitable, key=lambda g: g['hourly_cost'])

        # Rent instance
        try:
            response = self._api_request('PUT', f"asks/{offer['id']}/", {
                'client_id': 'learning_framework',
                'image': 'pytorch/pytorch:latest',
                'disk': 20
            })

            instance_id = response.get('new_contract')
            if not instance_id:
                raise ConnectionError("Failed to rent instance")

        except Exception as e:
            raise ConnectionError(f"Failed to rent instance: {e}")

        job_id = f"vast_{instance_id}"

        # Store job info
        self._active_jobs[job_id] = {
            'instance_id': instance_id,
            'offer': offer,
            'bundle_path': str(bundle_path),
            'config': asdict(job_config),
            'estimated_cost': estimated_cost,
            'status': 'starting',
            'start_time': time.time()
        }

        # TODO: In full implementation:
        # 1. Wait for SSH availability
        # 2. Upload bundle via SFTP
        # 3. Setup environment
        # 4. Start training in tmux

        return job_id

    def get_status(self, job_id: str) -> JobStatus:
        """Get current status of a job"""
        if job_id not in self._active_jobs:
            return JobStatus(
                job_id=job_id,
                status='unknown',
                message='Job not found'
            )

        job_info = self._active_jobs[job_id]
        elapsed = time.time() - job_info['start_time']

        # TODO: In full implementation:
        # 1. SSH into instance
        # 2. Check tmux session
        # 3. Parse training logs

        return JobStatus(
            job_id=job_id,
            status=job_info['status'],
            elapsed_time=elapsed,
            estimated_cost=job_info['estimated_cost']
        )

    def get_results(self, job_id: str) -> JobResult:
        """Get results from completed job"""
        if job_id not in self._active_jobs:
            return JobResult(
                job_id=job_id,
                success=False,
                error_message='Job not found'
            )

        job_info = self._active_jobs[job_id]

        # TODO: In full implementation:
        # 1. Download results.json via SCP
        # 2. Parse metrics
        # 3. Validate against baseline
        # 4. Destroy instance

        return JobResult(
            job_id=job_id,
            success=True,
            total_cost=job_info['estimated_cost']
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        if job_id not in self._active_jobs:
            return False

        job_info = self._active_jobs[job_id]
        instance_id = job_info['instance_id']

        try:
            self._api_request('DELETE', f"instances/{instance_id}/")
            job_info['status'] = 'cancelled'
            return True
        except Exception:
            return False
