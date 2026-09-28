import nest
import json
import os
import sys
import numpy as np

try:
    from tiago_ring_controller.math.circular import (
        angle_to_neuron_index,
        angle_to_ring_index,
        decode_population_angle,
        index_to_phase,
        preferred_angles,
        ring_index_to_angle,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tiago_ring_controller":
        raise
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tiago_ring_controller.math.circular import (
        angle_to_neuron_index,
        angle_to_ring_index,
        decode_population_angle,
        index_to_phase,
        preferred_angles,
        ring_index_to_angle,
    )
from tiago_ring_controller.math.ring import (
    builder_ring_distances,
    builder_ring_weight,
    generate_center_indices,
    stimulus_target_indices,
)
from tiago_ring_controller.config import load_neuron_parameters
from tiago_ring_controller.nest.ring import (
    build_ring_network,
    inject_stimulus as inject_ring_stimulus,
)


class RingAttractor:
    """Spiking neural ring attractor with Mexican-hat connectivity."""

    def __init__(self, population_size: int = 100, reset_kernel: bool = True,
                 params_file: str = None):
        """
        Create a ring attractor network.
        
        Args:
            population_size: Number of neurons
            reset_kernel: Reset NEST before building
            params_file: Path to neuron parameters JSON
        """
        if params_file is None:
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            params_file = os.path.join(script_dir, "config/model_params/neuron_params.json")
        
        if reset_kernel:
            nest.ResetKernel()
        nest.set_verbosity("M_ERROR")

        self.population_size = population_size
        self.max_distance = 50
        self.excitation_std_dev = 10
        self.inhibition_std_dev = 5

        self._load_neuron_params(params_file)
        self._build_network()

    def _load_neuron_params(self, params_file: str):
        """Load neuron parameters from JSON file."""
        self.neuron_params = load_neuron_parameters(params_file, "ring")

    def _build_network(self):
        """Create neurons, recorders, and recurrent connections."""
        network = build_ring_network(
            nest,
            self.population_size,
            self.neuron_params,
            variant="builder",
            reset_kernel=False,
            max_distance=self.max_distance,
            excitation_std_dev=self.excitation_std_dev,
            inhibition_std_dev=self.inhibition_std_dev,
            configure_backend=False,
            distance_function=self._compute_ring_distances,
            weight_function=self._compute_weight,
        )
        self.neurons = network.neurons
        self.spike_recorders = network.spike_recorders

    def _compute_ring_distances(self) -> np.ndarray:
        """Create symmetric distance distribution around the ring."""
        return builder_ring_distances(self.population_size, self.max_distance)

    def _compute_weight(self, distance: float) -> float:
        """
        Calculate synaptic weight using Mexican-hat profile.
        
        Positive weights (excitation) for near neurons,
        negative weights (inhibition) for far neurons.
        """
        return builder_ring_weight(
            distance,
            self.population_size,
            self.excitation_std_dev,
            self.inhibition_std_dev,
        )

    def inject_stimulus(self, center_index: int = 0, half_width: int = 5):
        """
        Activate neurons around center with brief Poisson stimulus (200Hz, 50ms).
        
        Args:
            center_index: Center neuron
            half_width: Neurons activated on each side
        """
        network = type("_BuilderRingView", (), {})()
        network.population_size = self.population_size
        network.neurons = self.neurons
        inject_ring_stimulus(
            nest,
            network,
            center_index=center_index,
            half_width=half_width,
        )

    def get_spike_counts(self) -> np.ndarray:
        """Return spike count for each neuron."""
        counts = []
        for recorder in self.spike_recorders:
            events = nest.GetStatus(recorder, "events")[0]
            spike_times = events.get("times", [])
            counts.append(len(spike_times))
        return np.array(counts, dtype=float)

    def get_spike_times(self, recorder):
        """Return spike times from one recorder."""
        events = nest.GetStatus(recorder, "events")[0]
        return events.get("times", [])

    def preferred_angles(self, n: int) -> np.ndarray:
        """Compute preferred angles for n neurons."""
        return preferred_angles(n)

    def index_to_phase(self, index: int) -> float:
        """Map neuron index to phase angle [0, 2π)."""
        return index_to_phase(index, self.population_size)

    def angle_to_neuron_index(self, angle: float) -> int:
        """Neuron index closest to angle."""
        return angle_to_neuron_index(angle, self.population_size)

    def angle_to_ring_index(self, angle: float, ring_size: int = None) -> float:
        """Ring position for angle."""
        size = self.population_size if ring_size is None else ring_size
        result = angle_to_ring_index(angle, size)
        return np.float64(result) if np.ndim(result) == 0 else result

    def ring_index_to_angle(self, index: float, ring_size: int = None) -> float:
        """Angle for ring position."""
        size = self.population_size if ring_size is None else ring_size
        result = ring_index_to_angle(index, size)
        return np.float64(result) if np.ndim(result) == 0 else result

    def decode_angle_from_spikes(self, spike_counts: np.ndarray) -> float:
        """Decode angle using population vector from spike counts."""
        counts = np.asarray(spike_counts, dtype=float)
        normalized = np.maximum(counts - np.min(counts), 0.0)
        angles = self.preferred_angles(len(counts))
        sin_sum = np.dot(normalized, np.sin(angles))
        cos_sum = np.dot(normalized, np.cos(angles))
        return float(np.arctan2(sin_sum, cos_sum))

    def generate_ring_positions(self, num_positions: int) -> np.ndarray:
        """Generate evenly spaced non-zero neuron indices around the ring."""
        return generate_center_indices(self.population_size, num_positions)

    def as_node_collection(self, nodes) -> nest.NodeCollection:
        """Convert nodes to NEST NodeCollection."""
        if isinstance(nodes, nest.NodeCollection):
            return nodes
        if isinstance(nodes[0], nest.NodeCollection):
            return nodes[0]
        return nest.NodeCollection([int(n) for n in nodes])
    
