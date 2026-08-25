from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import FlowSISBase
    from .set_mask import BasePrediction, InstancePrediction, SetMaskHead


def __getattr__(name: str) -> Any:
    if name == "FlowSISBase":
        from .base import FlowSISBase

        return FlowSISBase
    if name in {"BasePrediction", "InstancePrediction", "SetMaskHead"}:
        from .set_mask import BasePrediction, InstancePrediction, SetMaskHead

        return {
            "BasePrediction": BasePrediction,
            "InstancePrediction": InstancePrediction,
            "SetMaskHead": SetMaskHead,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FlowSISBase",
    "BasePrediction",
    "InstancePrediction",
    "SetMaskHead",
]
