"""TaskGain: research stub with the port contract only (plan section 5e, level 3)."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence, Tuple

from .base import Block, BuildContext, Connection, Port, spikes_in, spikes_out
from .params import TASK_GAIN_SCHEMA


class TaskGain(Block):
    """Bilinear gate: configuration features AND task-error direction → feedback onto joint belief rings.

    Not implemented: ``build`` raises.  The block exists so graphs can be drawn
    and validated with it and the editor shows its ports.
    """

    type_name: ClassVar[str] = "TaskGain"
    schema = TASK_GAIN_SCHEMA
    neural: ClassVar[bool] = True
    ports: ClassVar[Tuple[Port, ...]] = (
        spikes_in("configuration", "SignedProduct features of the belief rings"),
        spikes_in("left_in", "task-level decision: left"),
        spikes_in("right_in", "task-level decision: right"),
        spikes_out("feedback", "fitted feedback into a joint belief ring"),
    )

    def build(self, ctx: BuildContext, inputs: Mapping[str, Sequence[Connection]]) -> None:
        raise self.error("TaskGain is a research stub; its weights are not fitted yet (plan 5e, level 3)")


__all__ = ["TaskGain"]
