from imaginairy.samplers.base import SolverName  # noqa
from imaginairy.samplers.ddim import DDIMSolver

SOLVERS = [
    # PLMSSolver,
    DDIMSolver,
    # kdiff.DPMFastSampler,
    # kdiff.DPMAdaptiveSampler,
    # kdiff.LMSSampler,
    # kdiff.DPM2Sampler,
    # kdiff.DPM2AncestralSampler,
    # kdiff.DPMPP2MSampler,
    # kdiff.DPMPP2SAncestralSampler,
    # kdiff.EulerSampler,
    # kdiff.EulerAncestralSampler,
    # kdiff.HeunSampler,
]

SOLVER_LOOKUP = {s.short_name: s for s in SOLVERS}
SOLVER_TYPE_OPTIONS = [s.short_name for s in SOLVERS]
