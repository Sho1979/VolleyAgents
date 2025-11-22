from typing import List

from .event import Event


class Timeline:
    """
    Mantiene una lista globale di eventi ordinati nel tempo.
    Tutti gli agenti spingono qui i loro eventi.
    """

    def __init__(self):
        self.events: List[Event] = []

    def add(self, event: Event):
        self.events.append(event)

    def extend(self, events: List[Event]):
        self.events.extend(events)

    def sorted(self) -> List[Event]:
        return sorted(self.events, key=lambda e: e.time)

    def filter(self, type=None):
        if type is None:
            return self.sorted()
        return [e for e in self.sorted() if e.type == type]

    def window(self, t_start: float, t_end: float) -> List[Event]:
        return [e for e in self.sorted() if t_start <= e.time <= t_end]

    def __repr__(self):
        lines = [str(e) for e in self.sorted()]
        return "\n".join(lines)

