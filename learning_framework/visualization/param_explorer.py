"""Parameter exploration for interactive visualization"""

import numpy as np
from typing import Dict, List, Any, Callable, Optional


class ParameterExplorer:
    """Manages parameter definitions and exploration"""

    def __init__(self):
        self.parameters: Dict[str, Dict[str, Any]] = {}
        self.loss_function: Optional[Callable] = None

    def define_parameters(self, params: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Define exploration parameters

        Args:
            params: Dictionary of parameter definitions
                Each definition has: min, max, default, scale (linear/log), label

        Returns:
            Processed parameter dictionary
        """
        for name, config in params.items():
            self.parameters[name] = {
                'name': name,
                'min': config.get('min', 0),
                'max': config.get('max', 1),
                'value': config.get('default', config.get('min', 0)),
                'scale': config.get('scale', 'linear'),
                'label': config.get('label', name),
                'step': config.get('step', None)
            }

            # Compute step if not provided
            if self.parameters[name]['step'] is None:
                if self.parameters[name]['scale'] == 'log':
                    self.parameters[name]['step'] = 0.01
                else:
                    range_val = config.get('max', 1) - config.get('min', 0)
                    self.parameters[name]['step'] = range_val / 100

        return self.parameters

    def set_parameter(self, name: str, value: float):
        """Set parameter value

        Args:
            name: Parameter name
            value: New value
        """
        if name not in self.parameters:
            raise KeyError(f"Parameter '{name}' not defined")

        param = self.parameters[name]
        # Clamp to range
        value = max(param['min'], min(param['max'], value))
        self.parameters[name]['value'] = value

    def get_parameter(self, name: str) -> float:
        """Get parameter value

        Args:
            name: Parameter name

        Returns:
            Current parameter value
        """
        if name not in self.parameters:
            raise KeyError(f"Parameter '{name}' not defined")
        return self.parameters[name]['value']

    def get_all_parameters(self) -> Dict[str, float]:
        """Get all parameter values as dict"""
        return {name: p['value'] for name, p in self.parameters.items()}

    def set_loss_function(self, fn: Callable):
        """Set loss function for effect computation

        Args:
            fn: Function that takes param dict and returns loss
        """
        self.loss_function = fn

    def compute_effect(
        self,
        param_name: str,
        num_points: int = 50,
        other_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[float]]:
        """Compute effect of parameter on loss

        Args:
            param_name: Parameter to vary
            num_points: Number of points to sample
            other_params: Fixed values for other parameters

        Returns:
            Dictionary with 'values' and 'losses' lists
        """
        if self.loss_function is None:
            raise ValueError("Loss function not set")

        if param_name not in self.parameters:
            raise KeyError(f"Parameter '{param_name}' not defined")

        param = self.parameters[param_name]

        # Generate sample values
        if param['scale'] == 'log':
            values = np.logspace(
                np.log10(param['min']),
                np.log10(param['max']),
                num_points
            ).tolist()
        else:
            values = np.linspace(param['min'], param['max'], num_points).tolist()

        # Compute losses
        base_params = other_params or self.get_all_parameters()
        losses = []

        for val in values:
            test_params = base_params.copy()
            test_params[param_name] = val
            try:
                loss = self.loss_function(test_params)
                losses.append(float(loss))
            except Exception:
                losses.append(float('nan'))

        return {
            'param_name': param_name,
            'values': values,
            'losses': losses
        }

    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration for parameter controls

        Returns:
            Configuration for generating UI sliders
        """
        ui_params = []

        for name, param in self.parameters.items():
            ui_param = {
                'name': name,
                'label': param['label'],
                'min': param['min'],
                'max': param['max'],
                'value': param['value'],
                'step': param['step'],
                'scale': param['scale']
            }
            ui_params.append(ui_param)

        return {
            'parameters': ui_params
        }

    def to_json(self) -> Dict[str, Any]:
        """Serialize for API response"""
        return {
            'parameters': self.get_ui_config()['parameters'],
            'current_values': self.get_all_parameters()
        }
