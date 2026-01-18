import pytest
import numpy as np
from learning_framework.visualization.param_explorer import ParameterExplorer


def test_param_explorer_defines_parameters():
    """Test parameter explorer defines learnable parameters"""
    explorer = ParameterExplorer()

    params = explorer.define_parameters({
        'learning_rate': {'min': 0.001, 'max': 1.0, 'default': 0.01, 'scale': 'log'},
        'batch_size': {'min': 8, 'max': 256, 'default': 32, 'scale': 'linear'},
        'momentum': {'min': 0.0, 'max': 0.99, 'default': 0.9, 'scale': 'linear'}
    })

    assert 'learning_rate' in params
    assert params['learning_rate']['value'] == 0.01
    assert params['batch_size']['value'] == 32


def test_param_explorer_updates_parameter():
    """Test parameter explorer updates parameter values"""
    explorer = ParameterExplorer()

    explorer.define_parameters({
        'learning_rate': {'min': 0.001, 'max': 1.0, 'default': 0.01}
    })

    explorer.set_parameter('learning_rate', 0.1)
    assert explorer.get_parameter('learning_rate') == 0.1


def test_param_explorer_computes_effect():
    """Test parameter explorer can compute effect on loss"""
    explorer = ParameterExplorer()

    explorer.define_parameters({
        'learning_rate': {'min': 0.001, 'max': 1.0, 'default': 0.01}
    })

    # Define a simple loss function
    def loss_fn(params):
        lr = params['learning_rate']
        # Simulate: too low = high loss, optimal = low loss, too high = high loss
        return (np.log(lr) + 3) ** 2

    explorer.set_loss_function(loss_fn)

    # Compute effect across learning rate range
    effect_data = explorer.compute_effect('learning_rate', num_points=10)

    assert 'values' in effect_data
    assert 'losses' in effect_data
    assert len(effect_data['values']) == 10


def test_param_explorer_generates_ui_config():
    """Test parameter explorer generates UI configuration"""
    explorer = ParameterExplorer()

    explorer.define_parameters({
        'learning_rate': {
            'min': 0.001, 'max': 1.0, 'default': 0.01,
            'scale': 'log', 'label': 'Learning Rate'
        }
    })

    ui_config = explorer.get_ui_config()

    assert 'parameters' in ui_config
    assert ui_config['parameters'][0]['name'] == 'learning_rate'
    assert ui_config['parameters'][0]['label'] == 'Learning Rate'
