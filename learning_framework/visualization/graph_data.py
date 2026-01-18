"""Computational graph data generation for visualization"""

from typing import List, Dict, Any, Optional


class ComputationalGraphBuilder:
    """Builds graph data for visualization from network architecture"""

    def __init__(self):
        self.node_counter = 0

    def build_feedforward_graph(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activations: Optional[List[str]] = None,
        batch_size: int = 1,
        include_gradients: bool = False
    ) -> Dict[str, Any]:
        """Build graph for feedforward network

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activations: List of activation functions
            batch_size: Batch size for shape computation
            include_gradients: Include backward pass edges

        Returns:
            Graph data with nodes and edges
        """
        nodes = []
        edges = []

        # Default activations
        if activations is None:
            activations = ['relu'] * len(hidden_dims) + ['softmax']

        # Build layer dimensions
        all_dims = [input_dim] + hidden_dims + [output_dim]

        # Horizontal spacing
        x_spacing = 200
        y_center = 300

        prev_node_id = None
        prev_dim = input_dim

        for i, dim in enumerate(all_dims):
            # Layer node
            if i == 0:
                node_type = 'input'
                node_id = 'input'
                label = f'Input\n{batch_size}x{dim}'
            elif i == len(all_dims) - 1:
                node_type = 'output'
                node_id = 'output'
                label = f'Output\n{batch_size}x{dim}'
            else:
                node_type = 'hidden'
                node_id = f'hidden_{i-1}'
                label = f'Hidden {i-1}\n{batch_size}x{dim}'

            node = {
                'id': node_id,
                'type': node_type,
                'label': label,
                'shape': [batch_size, dim],
                'x': i * x_spacing * 2,
                'y': y_center
            }
            nodes.append(node)

            # Add operations between layers
            if prev_node_id is not None:
                # Matrix multiplication
                matmul_id = f'matmul_{i-1}'
                matmul_node = {
                    'id': matmul_id,
                    'type': 'matmul',
                    'label': f'W{i-1}\n{prev_dim}x{dim}',
                    'shape': [prev_dim, dim],
                    'x': (i * 2 - 1) * x_spacing,
                    'y': y_center - 50
                }
                nodes.append(matmul_node)

                # Edges: prev -> matmul -> current
                edges.append({
                    'source': prev_node_id,
                    'target': matmul_id,
                    'type': 'forward'
                })

                # Activation
                if i - 1 < len(activations):
                    act_name = activations[i - 1]
                    act_id = f'{act_name}_{i-1}'
                    act_node = {
                        'id': act_id,
                        'type': act_name,
                        'label': act_name.upper(),
                        'x': (i * 2 - 0.5) * x_spacing,
                        'y': y_center + 50
                    }
                    nodes.append(act_node)

                    edges.append({
                        'source': matmul_id,
                        'target': act_id,
                        'type': 'forward'
                    })
                    edges.append({
                        'source': act_id,
                        'target': node_id,
                        'type': 'forward'
                    })
                else:
                    edges.append({
                        'source': matmul_id,
                        'target': node_id,
                        'type': 'forward'
                    })

            prev_node_id = node_id
            prev_dim = dim

        # Add gradient edges (backward pass)
        if include_gradients:
            grad_edges = self._build_gradient_edges(edges)
            edges.extend(grad_edges)

        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden_dims': hidden_dims,
                'batch_size': batch_size
            }
        }

    def _build_gradient_edges(self, forward_edges: List[Dict]) -> List[Dict]:
        """Build gradient (backward) edges from forward edges"""
        grad_edges = []

        for edge in forward_edges:
            if edge['type'] == 'forward':
                grad_edges.append({
                    'source': edge['target'],
                    'target': edge['source'],
                    'type': 'gradient'
                })

        return grad_edges

    def build_from_dezero(self, variables) -> Dict[str, Any]:
        """Build graph from dezero computational graph

        Args:
            variables: List of dezero Variable objects

        Returns:
            Graph data for visualization
        """
        # Topological sort of computation graph
        nodes = []
        edges = []
        seen = set()

        def add_func(f):
            if f is None or id(f) in seen:
                return
            seen.add(id(f))

            # Add function node
            node = {
                'id': f'func_{id(f)}',
                'type': f.__class__.__name__,
                'label': f.__class__.__name__
            }
            nodes.append(node)

            # Add edges from inputs
            for x in f.inputs:
                edge = {
                    'source': f'var_{id(x)}',
                    'target': f'func_{id(f)}',
                    'type': 'forward'
                }
                edges.append(edge)
                add_var(x)

            # Add edges to outputs
            for y in f.outputs:
                y_ref = y()  # Weak reference
                if y_ref:
                    edge = {
                        'source': f'func_{id(f)}',
                        'target': f'var_{id(y_ref)}',
                        'type': 'forward'
                    }
                    edges.append(edge)

        def add_var(v):
            if id(v) in seen:
                return
            seen.add(id(v))

            shape_str = 'x'.join(str(d) for d in v.shape) if v.data is not None else '?'
            node = {
                'id': f'var_{id(v)}',
                'type': 'variable',
                'label': f'{v.name or "var"}\n{shape_str}',
                'shape': list(v.shape) if v.data is not None else None
            }
            nodes.append(node)

            if v.creator is not None:
                add_func(v.creator)

        # Start from output variables
        for v in variables:
            add_var(v)

        return {
            'nodes': nodes,
            'edges': edges
        }
