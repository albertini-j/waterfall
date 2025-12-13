"""Biblioteca simples para planejamento de projetos waterfall.

Este módulo é autocontido: basta colocar o arquivo na pasta do seu projeto
para usar ``import waterfall as wf`` sem instalação de pacote.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
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
        Cargas padrão (legado). Se você quiser nomes e quantidades customizadas,
        use o dicionário ``resources`` ao criar a atividade.
    resources:
        Dicionário opcional ``{nome_recurso: carga}`` para atribuição explícita.
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
    resource1: float = 0.0
    resource2: float = 0.0
    resource3: float = 0.0
    resources: Optional[Dict[str, float]] = None
    delay: float = 0.0
    predecessors: List[str] = field(default_factory=list)
    start: Optional[datetime] = None
    finish: Optional[datetime] = None
    early_start: Optional[datetime] = None
    late_start: Optional[datetime] = None
    total_float: Optional[float] = None
    is_critical: bool = False

    def __post_init__(self) -> None:
        # Permite uso legado (resource1..3) e customizado (dict "resources").
        if self.resources is None:
            self.resources = {
                "resource1": self.resource1,
                "resource2": self.resource2,
                "resource3": self.resource3,
            }
        else:
            # Cria uma cópia para evitar mutação externa.
            self.resources = dict(self.resources)

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

    def __init__(
        self,
        start_date: Optional[datetime] = None,
        *,
        resource_count: int = 3,
        resource_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.start_date: datetime = start_date or datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self.activities: Dict[str, Activity] = {}
        if resource_names is not None:
            self.resource_names: List[str] = list(resource_names)
        else:
            if resource_count < 1:
                raise ValueError("resource_count deve ser pelo menos 1")
            self.resource_names = [f"resource{i+1}" for i in range(resource_count)]
        self.resource_histogram: List[Dict[str, object]] = []

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

    def _build_resource_histogram(self) -> List[Dict[str, object]]:
        if not self.activities:
            return []

        usage: Dict[datetime, Dict[str, float]] = defaultdict(
            lambda: {name: 0.0 for name in self.resource_names}
        )

        for activity in self.activities.values():
            if activity.start is None or activity.finish is None:
                raise ScheduleError("Atualize o cronograma antes de gerar o histograma de recursos.")

            unknown = set(activity.resources or {}).difference(self.resource_names)
            if unknown:
                raise ScheduleError(
                    f"Atividade {activity.activity_id} possui recursos não configurados: {', '.join(sorted(unknown))}."
                )

            current = activity.start
            while current < activity.finish:
                end_of_day = datetime.combine(current.date(), time.min) + timedelta(days=1)
                slice_finish = min(end_of_day, activity.finish)
                portion_days = (slice_finish - current).total_seconds() / 86400

                day_key = datetime.combine(current.date(), time.min)
                for name in self.resource_names:
                    usage[day_key][name] += (activity.resources or {}).get(name, 0.0) * portion_days

                current = slice_finish

        histogram: List[Dict[str, object]] = []
        for day in sorted(usage.keys()):
            per_resource = usage[day]
            histogram.append({
                "date": day,
                "resources": per_resource,
                "total": sum(per_resource.values()),
            })

        return histogram

    def plot_resource_histogram(
        self,
        *,
        title: str = "Histograma de Recursos",
        resources: Optional[Sequence[str]] = None,
    ):
        """
        Gera um histograma por recurso configurado.

        Como usar:
        - Após chamar ``update_schedule``, invoque ``plot_resource_histogram()``
          para produzir um gráfico separado para cada recurso configurado.
        - Use ``resources=["pedreiros", "eletricistas"]`` para filtrar e
          renderizar apenas alguns recursos. Quando ``resources`` é ``None``,
          todos os recursos definidos em ``ProjectSchedule.resource_names`` são
          exibidos.
        - Cada subplot mostra barras da carga diária do recurso respectivo.
        """

        if not self.resource_histogram:
            raise ScheduleError("Nenhum histograma disponível. Rode update_schedule primeiro.")

        selected = list(resources) if resources is not None else list(self.resource_names)
        unknown = set(selected).difference(self.resource_names)
        if unknown:
            raise ScheduleError(
                f"Recursos não configurados para plotagem: {', '.join(sorted(unknown))}."
            )

        dates = [slot["date"] for slot in self.resource_histogram]

        fig, axes = plt.subplots(
            nrows=len(selected),
            ncols=1,
            sharex=True,
            figsize=(10, max(3, 2 * len(selected))),
        )
        if len(selected) == 1:
            axes = [axes]

        for ax, resource_name in zip(axes, selected):
            values = [slot["resources"].get(resource_name, 0.0) for slot in self.resource_histogram]
            ax.bar(dates, values, label=resource_name)
            ax.set_ylabel("Carga")
            ax.legend()

        axes[0].set_title(title)
        axes[-1].set_xlabel("Data")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig, axes

    def get_activity(self, activity_id: str) -> Activity:
        return self.activities[activity_id]

    def find_by_activity_id(self, activity_id: str) -> Activity:
        """Alias explícito para recuperar uma atividade pelo ID."""

        return self.get_activity(activity_id)

    def find_by_name(self, name: str) -> List[Activity]:
        """Lista atividades com nome exatamente igual (case-insensitive)."""

        return [a for a in self.activities.values() if a.name.lower() == name.lower()]

    def find_by_area(self, area: str) -> List[Activity]:
        """Retorna todas as atividades de uma determinada área (case-insensitive)."""

        return [a for a in self.activities.values() if a.area.lower() == area.lower()]

    def activities_on_date(self, target_date: date) -> List[Activity]:
        """Retorna atividades que ocorrem (mesmo parcialmente) no dia informado."""

        if any(a.start is None or a.finish is None for a in self.activities.values()):
            raise ScheduleError("Atualize o cronograma antes de consultar por datas.")

        day_start = datetime.combine(target_date, time.min)
        day_end = day_start + timedelta(days=1)
        return [
            a
            for a in self.activities.values()
            if a.start < day_end and a.finish > day_start  # type: ignore[operator]
        ]

    def activities_in_period(self, start: date, end: date) -> List[Activity]:
        """Retorna atividades que intersectam o período [start, end]."""

        if end < start:
            raise ValueError("O fim do período deve ser igual ou posterior ao início.")

        if any(a.start is None or a.finish is None for a in self.activities.values()):
            raise ScheduleError("Atualize o cronograma antes de consultar por datas.")

        start_dt = datetime.combine(start, time.min)
        end_dt = datetime.combine(end, time.min) + timedelta(days=1)
        return [
            a
            for a in self.activities.values()
            if a.start < end_dt and a.finish > start_dt  # type: ignore[operator]
        ]

    def reset(self) -> None:
        for activity in self.activities.values():
            activity.clear_schedule()
        self.activities.clear()

