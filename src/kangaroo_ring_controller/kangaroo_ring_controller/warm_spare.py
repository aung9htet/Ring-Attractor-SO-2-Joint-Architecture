"""The warm spare: a ring built in advance for the selected joint, so the next reach starts without the 22-30 s build
(CONTEXT.md; DECISIONS.md D30, D37). Holds at most one ring; NEST-free itself, the builder is passed in.
"""


class WarmSpare:
    def __init__(self, build):
        """`build(selected) -> ring`, selected = (side, joint, seed); the ring carries `.setup` and `.seed`."""
        self.build = build
        self.selected, self.ring = None, None

    def due(self, selected):
        """Whether a ring should be built for `selected` now."""
        return selected != self.selected

    def prepare(self, selected):
        """Build the ring for `selected` now. If the build raises, the error propagates and is not retried until the
        selection changes."""
        self.selected, self.ring = selected, None
        self.ring = self.build(selected)

    def take(self, setup, seed):
        """The ring if it was built for `setup` and `seed` (D35), else None; used up either way."""
        ring, self.selected, self.ring = self.ring, None, None  # used up: the next idle pass builds another
        return ring if ring and (ring.setup, ring.seed) == (setup, seed) else None
