# Simulation package — import submodules to trigger @register_* decorators
import drone_sim.simulation.centralized.coordinator  # noqa: F401
import drone_sim.simulation.distributed.distributed_coordinator  # noqa: F401
import drone_sim.simulation.distributed.threaded_coordinator  # noqa: F401
import drone_sim.simulation.centralized.conflict_evasion_coordinator  # noqa: F401
import drone_sim.simulation.distributed.conflict_evasion_coordinator  # noqa: F401
import drone_sim.simulation.intersection.intersection_dmpc_coordinator  # noqa: F401