"""
ANC-SIH pipeline package — Phase 3.

Exports
-------
HybridANC          — complete streaming noise-cancellation engine
NLMSPostFilter     — adaptive NLMS post-filter with DTD
ImpulseHandler     — burst suppressor (from Phase 2)
FrameTrace         — per-hop trace record
PipelineTrace      — aggregated trace for a full file
"""
from anc.pipeline.hybrid import HybridANC
from anc.pipeline.nlms_postfilter import NLMSPostFilter, NLMSPostFilterConfig
from anc.pipeline.impulse_handler import ImpulseHandler, ImpulseHandlerConfig
from anc.pipeline.trace import FrameTrace, PipelineTrace

__all__ = [
    "HybridANC",
    "NLMSPostFilter",
    "NLMSPostFilterConfig",
    "ImpulseHandler",
    "ImpulseHandlerConfig",
    "FrameTrace",
    "PipelineTrace",
]
