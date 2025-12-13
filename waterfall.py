"""Biblioteca simples para planejamento de projetos waterfall.

Este módulo é autocontido: basta colocar o arquivo na pasta do seu projeto
para usar ``import waterfall as wf`` sem instalação de pacote.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, Iterable, List, Optional, Sequence
import warnings

import matplotlib.pyplot as plt

__all__ = [
    "Activity",
    "ProjectSchedule",
    "ScheduleError",
    "DuplicateActivityError",
    "CycleError",
    "UnknownPredecessorError",
]


@dataclass
class Activity:
    """Representa uma atividade em um cronograma waterfall.

    Atributos
    ---------
    name:
        Nome legível da atividade.
    activity_id:
        Identificador único para referência e ordenação.
    area:
        Domínio ou equipe responsável.
    short_description:
        Resumo da atividade.
    long_description:
        Descrição detalhada.
    duration:
        Duração em dias (aceita valores fracionários).
    resource1, resource2, resource3:
        Esforço ou carga associada a cada recurso.
    delay:
        Atraso aplicado ao início da atividade (em dias). Não bloqueia o
        planejamento, mas pode reduzir a folga.
    predecessors:
        Lista de IDs de atividades que devem terminar antes desta iniciar.
    start:
        Data de início calculada após o planejamento.
    finish:
        Data de término calculada após o planejamento.
    early_start / late_start:
        Datas de início mais cedo e mais tarde considerando as dependências.
    total_float:
        Folga total (late_start - early_start) em dias.
    is_critical:
        Indica se a atividade está no caminho crítico (folga zero).
    """

    name: str
    activity_id: str
    area: str
    short_description: str
    long_description: str
    duration: float
    resource1: float
    resource2: float
    resource3: float
    delay: float = 0.0
    predecessors: List[str] = field(default_factory=list)
    start: Optional[datetime] = None
    finish: Optional[datetime] = None
    early_start: Optional[datetime] = None
    late_start: Optional[datetime] = None
    total_float: Optional[float] = None
    is_critical: bool = False

    def set_schedule(self, start: datetime) -> None:
        """Ajusta datas de início e fim com base na duração configurada."""

        self.start = start
        self.early_start = start
        self.finish = start + timedelta(days=self.duration)

    def depends_on(self, candidates: Iterable[str]) -> bool:
        """Retorna se a atividade depende de pelo menos um ID da lista."""

        predecessors = set(self.predecessors)
        return any(candidate in predecessors for candidate in candidates)

    def clear_schedule(self) -> None:
        """Remove informações de datas calculadas."""

        self.start = None
        self.finish = None
        self.early_start = None
        self.late_start = None
        self.total_float = None
        self.is_critical = False


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
        self.start_date: datetime = start_date or datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self.activities: Dict[str, Activity] = {}
        self.resource_histogram: List[tuple[datetime, float, float, float, float]] = []

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

    def update_schedule(
        self,
        plot: bool = False,
        title: str = "Cronograma",
        *,
        plot_resources: bool = False,
        resource_title: str = "Histograma de Recursos",
    ):
        """Calcula datas de início/fim e opcionalmente plota gráficos de Gantt e recursos."""

        self._validate_dependencies()
        order = self._topological_order()

        successors: Dict[str, List[str]] = defaultdict(list)
        for activity in self.activities.values():
            for predecessor in activity.predecessors:
                successors[predecessor].append(activity.activity_id)

        for activity in self.activities.values():
            activity.clear_schedule()

        for activity_id in order:
            activity = self.activities[activity_id]
            if activity.predecessors:
                latest_finish = max(self.activities[pred].finish for pred in activity.predecessors)
                start = latest_finish
            else:
                start = self.start_date

            start_with_delay = start + timedelta(days=activity.delay)
            activity.set_schedule(start_with_delay)

        project_finish = max(activity.finish for activity in self.activities.values()) if self.activities else self.start_date

        for activity_id in reversed(order):
            activity = self.activities[activity_id]

            if successors.get(activity_id):
                successor_starts = [self.activities[suc].late_start for suc in successors[activity_id] if self.activities[suc].late_start is not None]
                late_finish = min(successor_starts) if successor_starts else project_finish
            else:
                late_finish = project_finish

            activity.late_start = late_finish - timedelta(days=activity.duration)
            if activity.early_start is None:
                raise ScheduleError("early_start não calculado para a atividade.")

            total_float = (activity.late_start - activity.early_start).total_seconds() / 86400
            activity.total_float = total_float
            activity.is_critical = abs(total_float) < 1e-9

            if activity.delay > total_float:
                warnings.warn(
                    (
                        f"Delay da atividade {activity.activity_id} ({activity.delay} dias) "
                        f"maior que a folga total calculada ({total_float} dias)."
                    ),
                    RuntimeWarning,
                )

        self.resource_histogram = self._build_resource_histogram()

        if plot:
            gantt = self.plot_gantt(title=title)
        else:
            gantt = None

        if plot_resources:
            histogram = self.plot_resource_histogram(title=resource_title)
        else:
            histogram = None

        if plot or plot_resources:
            return gantt if histogram is None else (gantt, histogram)

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

    def _build_resource_histogram(self) -> List[tuple[datetime, float, float, float, float]]:
        if not self.activities:
            return []

        usage: Dict[datetime, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

        for activity in self.activities.values():
            if activity.start is None or activity.finish is None:
                raise ScheduleError("Atualize o cronograma antes de gerar o histograma de recursos.")

            current = activity.start
            while current < activity.finish:
                end_of_day = datetime.combine(current.date(), time.min) + timedelta(days=1)
                slice_finish = min(end_of_day, activity.finish)
                portion_days = (slice_finish - current).total_seconds() / 86400

                day_key = datetime.combine(current.date(), time.min)
                usage[day_key][0] += activity.resource1 * portion_days
                usage[day_key][1] += activity.resource2 * portion_days
                usage[day_key][2] += activity.resource3 * portion_days

                current = slice_finish

        histogram: List[tuple[datetime, float, float, float, float]] = []
        for day in sorted(usage.keys()):
            r1, r2, r3 = usage[day]
            histogram.append((day, r1, r2, r3, r1 + r2 + r3))

        return histogram

    def plot_resource_histogram(self, *, title: str = "Histograma de Recursos"):
        if not self.resource_histogram:
            raise ScheduleError("Nenhum histograma disponível. Rode update_schedule primeiro.")

        dates = [slot[0] for slot in self.resource_histogram]
        r1 = [slot[1] for slot in self.resource_histogram]
        r2 = [slot[2] for slot in self.resource_histogram]
        r3 = [slot[3] for slot in self.resource_histogram]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(dates, r1, label="Recurso 1")
        ax.bar(dates, r2, bottom=r1, label="Recurso 2")
        bottom_r3 = [v1 + v2 for v1, v2 in zip(r1, r2)]
        ax.bar(dates, r3, bottom=bottom_r3, label="Recurso 3")

        ax.set_title(title)
        ax.set_ylabel("Carga")
        ax.set_xlabel("Data")
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig, ax

    def get_activity(self, activity_id: str) -> Activity:
        return self.activities[activity_id]

    def reset(self) -> None:
        for activity in self.activities.values():
            activity.clear_schedule()
        self.activities.clear()

