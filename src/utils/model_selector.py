"""
Model selection utility for optimal performance and confidence.

This module helps select the best LLM model based on available hardware,
performance requirements, and confidence needs.
"""

import psutil
import subprocess
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelSpec:
    """Specification for a model including resource requirements."""

    name: str
    min_ram_gb: float
    recommended_ram_gb: float
    min_vram_gb: Optional[float] = None
    recommended_vram_gb: Optional[float] = None
    context_length: int = 4096
    expected_confidence: float = 0.7
    speed_rating: int = 5  # 1-10 scale


class ModelSelector:
    """Utility for selecting optimal models based on system capabilities."""

    # Model specifications based on typical performance
    MODEL_SPECS = {
        "gemma3:1b": ModelSpec(
            name="gemma3:1b",
            min_ram_gb=4.0,
            recommended_ram_gb=8.0,
            context_length=8192,
            expected_confidence=0.6,
            speed_rating=8,
        ),
        "gemma3:4b": ModelSpec(
            name="gemma3:4b",
            min_ram_gb=8.0,
            recommended_ram_gb=16.0,
            context_length=8192,
            expected_confidence=0.7,
            speed_rating=6,
        ),
        "llama3.1:8b": ModelSpec(
            name="llama3.1:8b",
            min_ram_gb=16.0,
            recommended_ram_gb=32.0,
            context_length=8192,
            expected_confidence=0.8,
            speed_rating=5,
        ),
        "llama3.1:70b": ModelSpec(
            name="llama3.1:70b",
            min_ram_gb=64.0,
            recommended_ram_gb=128.0,
            context_length=8192,
            expected_confidence=0.9,
            speed_rating=2,
        ),
        "mistral:7b": ModelSpec(
            name="mistral:7b",
            min_ram_gb=12.0,
            recommended_ram_gb=24.0,
            context_length=8192,
            expected_confidence=0.75,
            speed_rating=4,
        ),
        "qwen2.5:7b": ModelSpec(
            name="qwen2.5:7b",
            min_ram_gb=12.0,
            recommended_ram_gb=24.0,
            context_length=32768,
            expected_confidence=0.8,
            speed_rating=4,
        ),
    }

    def __init__(self):
        """Initialize the model selector."""
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict:
        """Get system information for model selection."""
        return {
            "total_ram_gb": psutil.virtual_memory().total / (1024**3),
            "available_ram_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        }

    def _check_ollama_models(self) -> List[str]:
        """Check which models are available in Ollama."""
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                return [line.split()[0] for line in lines if line.strip()]
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.SubprocessError,
        ):
            pass
        return []

    def get_recommended_models(
        self, prioritize_confidence: bool = True, max_ram_usage: Optional[float] = None
    ) -> List[ModelSpec]:
        """Get recommended models based on system capabilities.

        Args:
            prioritize_confidence: If True, prioritize models with higher confidence.
            max_ram_usage: Maximum RAM usage allowed (GB). If None, uses available RAM.

        Returns:
            List of recommended models sorted by preference.
        """
        available_models = self._check_ollama_models()
        system_ram = max_ram_usage or self.system_info["available_ram_gb"]

        # Filter models that can run on this system
        compatible_models = []
        for model_name, spec in self.MODEL_SPECS.items():
            if spec.min_ram_gb <= system_ram:
                # Check if model is available
                is_available = any(
                    available_model.startswith(model_name.split(":")[0])
                    for available_model in available_models
                )
                if is_available:
                    compatible_models.append(spec)

        # Sort by preference
        if prioritize_confidence:
            compatible_models.sort(
                key=lambda x: (x.expected_confidence, x.speed_rating), reverse=True
            )
        else:
            compatible_models.sort(
                key=lambda x: (x.speed_rating, x.expected_confidence), reverse=True
            )

        return compatible_models

    def get_best_model(
        self, prioritize_confidence: bool = True, max_ram_usage: Optional[float] = None
    ) -> Optional[ModelSpec]:
        """Get the best model for the current system.

        Args:
            prioritize_confidence: If True, prioritize models with higher confidence.
            max_ram_usage: Maximum RAM usage allowed (GB).

        Returns:
            Best model specification or None if no compatible models.
        """
        recommended = self.get_recommended_models(prioritize_confidence, max_ram_usage)
        return recommended[0] if recommended else None

    def get_system_report(self) -> Dict:
        """Get a detailed system report for debugging."""
        return {
            "system_info": self.system_info,
            "available_models": self._check_ollama_models(),
            "recommended_for_confidence": [
                m.name for m in self.get_recommended_models(prioritize_confidence=True)
            ],
            "recommended_for_speed": [
                m.name for m in self.get_recommended_models(prioritize_confidence=False)
            ],
        }


def get_optimal_model_config(prioritize_confidence: bool = True) -> Dict:
    """Get optimal model configuration for the current system.

    Args:
        prioritize_confidence: If True, prioritize models with higher confidence.

    Returns:
        Dictionary with model configuration.
    """
    selector = ModelSelector()
    best_model = selector.get_best_model(prioritize_confidence)

    if best_model:
        return {
            "provider": "ollama",
            "model": best_model.name,
            "expected_confidence": best_model.expected_confidence,
            "context_length": best_model.context_length,
        }
    else:
        # Fallback to gemma3:4b if no compatible models found
        return {
            "provider": "ollama",
            "model": "gemma3:4b",
            "expected_confidence": 0.7,
            "context_length": 8192,
        }


if __name__ == "__main__":
    """CLI interface for model selection."""
    import argparse

    parser = argparse.ArgumentParser(description="Model selection utility")
    parser.add_argument(
        "--confidence", action="store_true", help="Prioritize confidence over speed"
    )
    parser.add_argument(
        "--speed", action="store_true", help="Prioritize speed over confidence"
    )
    parser.add_argument(
        "--report", action="store_true", help="Show detailed system report"
    )

    args = parser.parse_args()

    selector = ModelSelector()

    if args.report:
        import json

        print(json.dumps(selector.get_system_report(), indent=2))
    else:
        prioritize_confidence = args.confidence or not args.speed
        best_model = selector.get_best_model(prioritize_confidence)

        if best_model:
            print(f"Recommended model: {best_model.name}")
            print(f"Expected confidence: {best_model.expected_confidence:.2f}")
            print(f"Speed rating: {best_model.speed_rating}/10")
            print(f"RAM requirement: {best_model.recommended_ram_gb:.1f} GB")
        else:
            print("No compatible models found!")
            print(
                "Available system RAM:",
                f"{selector.system_info['available_ram_gb']:.1f} GB",
            )
