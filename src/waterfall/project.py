from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt

from .activity import Activity


class ScheduleError(Exception):
    """Erro genérico de planejamento."""


class DuplicateActivityError(ScheduleError):
    """ID de atividade duplicado."""


class CycleError(ScheduleError):
    """Detecta dependências cíclicas que impedem o cronograma."""


class UnknownPredecessorError(ScheduleError):
    """Lançado quando uma dependência não é encontrada na lista de atividades."""


class ProjectSchedule:
    """Mantém a lista de atividades e calcula datas de início e fim."""

    def __init__(self, start_date: Optional[datetime] = None) -> None:
        self.start_date: datetime = start_date or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.activities: Dict[str, Activity] = {}

    def add_activity(self, activity: Activity) -> None:
        if activity.activity_id in self.activities:
            raise DuplicateActivityError(f"ID duplicado: {activity.activity_id}")
        self.activities[activity.activity_id] = activity

    def add_activities(self, activities: Iterable[Activity]) -> None:
        for activity in activities:
            self.add_activity(activity)

    def _validate_dependencies(self) -> None:
        missing: Dict[str, List[str]] = defaultdict(list)
        for activity in self.activities.values():
            for pred in activity.predecessors:
                if pred not in self.activities:
                    missing[activity.activity_id].append(pred)
        if missing:
            lines = [f"{aid} -> {', '.join(sorted(preds))}" for aid, preds in sorted(missing.items())]
            raise UnknownPredecessorError("Dependências não encontradas: " + "; ".join(lines))

    def _topological_order(self) -> List[str]:
        in_degree: Dict[str, int] = defaultdict(int)
        graph: Dict[str, List[str]] = defaultdict(list)

        for activity in self.activities.values():
            in_degree.setdefault(activity.activity_id, 0)
            for predecessor in activity.predecessors:
                graph[predecessor].append(activity.activity_id)
                in_degree[activity.activity_id] += 1

        queue: deque[str] = deque([aid for aid, degree in in_degree.items() if degree == 0])
        order: List[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.activities):
            raise CycleError("Não foi possível ordenar as atividades devido a um ciclo.")

        return order

    def update_schedule(self, plot: bool = False, title: str = "Cronograma"):
        """Calcula datas de início e fim e opcionalmente plota o gráfico de Gantt."""

        self._validate_dependencies()
        order = self._topological_order()

        for activity in self.activities.values():
            activity.clear_schedule()

        for activity_id in order:
            activity = self.activities[activity_id]
            if activity.predecessors:
                latest_finish = max(self.activities[pred].finish for pred in activity.predecessors)
                start = latest_finish
            else:
                start = self.start_date

            activity.set_schedule(start)

        if plot:
            return self.plot_gantt(title=title)

    def plot_gantt(self, *, title: str = "Cronograma"):
        if not self.activities:
            raise ScheduleError("Nenhuma atividade cadastrada para gerar o gráfico.")

        if any(activity.start is None or activity.finish is None for activity in self.activities.values()):
            raise ScheduleError("Atualize o cronograma antes de plotar o gráfico.")

        sorted_items: Sequence[Activity] = sorted(self.activities.values(), key=lambda a: a.start or self.start_date)

        fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_items) * 0.6)))
        yticks: List[int] = []
        ylabels: List[str] = []

        for index, activity in enumerate(sorted_items):
            start = activity.start or self.start_date
            duration: timedelta = (activity.finish or start) - start
            ax.barh(index, duration.days + duration.seconds / 86400, left=start, align="center")
            ax.text(start, index, activity.name, va="center", ha="right")
            yticks.append(index)
            ylabels.append(f"{activity.activity_id} ({activity.area})")

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Data")
        ax.set_title(title)
        plt.tight_layout()
        return fig, ax

    def get_activity(self, activity_id: str) -> Activity:
        return self.activities[activity_id]

    def reset(self) -> None:
        for activity in self.activities.values():
            activity.clear_schedule()
        self.activities.clear()

