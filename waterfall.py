"""Lightweight waterfall planning utilities.

This module is self-contained: drop the file into your project folder to
``import waterfall as wf`` without installing a package.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import date, datetime, time, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "Activity",
    "ProjectSchedule",
    "ScheduleError",
    "DuplicateActivityError",
    "CycleError",
    "UnknownPredecessorError",
]


class Activity(BaseModel):
    """Represents a task in the waterfall schedule.

    Use this class to describe each project task, specifying duration,
    area, descriptions, dependencies, and resource allocation through
    the ``resources`` dictionary (for example ``{"civil_engineers": 2}``).
    The ``weight`` attribute contributes to S-curve calculations. Use
    ``progress_percent`` to track earned progress; once it reaches 100%,
    you may also record ``actual_finish``.
    The ``delay`` field lets you intentionally postpone the start without
    blocking scheduling.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    activity_id: str
    area: str
    short_description: str
    long_description: str
    duration: float
    weight: float = 1.0
    progress_percent: float = 0.0
    actual_finish: Optional[datetime] = None
    resources: Dict[str, float] = Field(default_factory=dict)
    delay: float = 0.0
    predecessors: List[str] = Field(default_factory=list)
    start: Optional[datetime] = None
    finish: Optional[datetime] = None
    early_start: Optional[datetime] = None
    late_start: Optional[datetime] = None
    total_float: Optional[float] = None
    is_critical: bool = False

    @field_validator("progress_percent")
    @classmethod
    def validate_progress_percent(cls, value: float) -> float:
        """Ensure progress stays between 0 and 100 percent."""

        if value < 0 or value > 100:
            raise ValueError("progress_percent must be between 0 and 100.")
        return value

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        """Ensure weight is positive so progress curves can be computed."""

        if value <= 0:
            raise ValueError("weight must be greater than zero.")
        return value

    @model_validator(mode="after")
    def clone_resources(self) -> "Activity":
        """Clone the resources dictionary to avoid external mutation."""

        self.resources = dict(self.resources)
        return self

    @model_validator(mode="after")
    def validate_progress_dates(self) -> "Activity":
        """Normalize progress completion dates and enforce consistency rules."""

        if self.actual_finish is not None and self.progress_percent < 100:
            raise ValueError("actual_finish can only be set when progress_percent is 100%.")

        if self.actual_finish is not None:
            self.actual_finish = self.actual_finish.replace(hour=0, minute=0, second=0, microsecond=0)

        return self

    def set_schedule(self, start: datetime) -> None:
        """Define start/finish and early start for internal scheduling."""

        self.start = start
        self.early_start = start
        self.finish = start + timedelta(days=self.duration)

    def depends_on(self, candidates: Iterable[str]) -> bool:
        """Return True when the activity depends on any provided IDs."""

        predecessors = set(self.predecessors)
        return any(candidate in predecessors for candidate in candidates)

    def clear_schedule(self) -> None:
        """Reset all computed dates and float data prior to recalculation."""

        self.start = None
        self.finish = None
        self.early_start = None
        self.late_start = None
        self.total_float = None
        self.is_critical = False


class ScheduleError(Exception):
    """Generic scheduling error."""


class DuplicateActivityError(ScheduleError):
    """Raised when an activity ID is duplicated."""


class CycleError(ScheduleError):
    """Raised when cyclic dependencies prevent scheduling."""


class UnknownPredecessorError(ScheduleError):
    """Raised when a declared predecessor is missing."""


def _default_start_date() -> datetime:
    """Generate a normalized midnight start date for defaults."""

    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


class ProjectSchedule(BaseModel):
    """Stores activities, computes dates, and offers plotting/query helpers.

    Besides scheduling (early/late dates, total float, critical path) it
    generates Gantt charts, resource histograms, and planned vs. actual
    S-curves based on activity weights and reported progress. The schedule
    keeps a single ``progress_as_of`` date that applies to all activities
    when building the actual S-curve.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start_date: datetime = Field(default_factory=_default_start_date)
    resource_count: int = 3
    resource_names: List[str] = Field(default_factory=list)
    activities: Dict[str, Activity] = Field(default_factory=dict)
    resource_histogram: List[Dict[str, object]] = Field(default_factory=list)
    progress_as_of: Optional[datetime] = None

    @field_validator("start_date")
    @classmethod
    def normalize_start_date(cls, value: datetime) -> datetime:
        """Force the start date to midnight for consistent calculations."""

        return value.replace(hour=0, minute=0, second=0, microsecond=0)

    @model_validator(mode="after")
    def populate_resource_names(self) -> "ProjectSchedule":
        """Populate resource names based on the provided count when absent."""

        if not self.resource_names:
            if self.resource_count < 1:
                raise ValueError("resource_count must be at least 1")
            self.resource_names = [f"resource{i+1}" for i in range(self.resource_count)]
        return self

    @field_validator("progress_as_of")
    @classmethod
    def normalize_progress_as_of(cls, value: Optional[datetime]) -> Optional[datetime]:
        """Normalize the progress reference date to midnight when present."""

        if value is None:
            return None
        return value.replace(hour=0, minute=0, second=0, microsecond=0)

    def _total_weight(self) -> float:
        """Return the sum of all activity weights, defaulting to 0.0 when empty."""

        return sum(activity.weight for activity in self.activities.values())

    def _ensure_schedule_ready(self) -> None:
        """Raise when schedule-dependent data has not been computed."""

        if any(activity.start is None or activity.finish is None for activity in self.activities.values()):
            raise ScheduleError("Run update_schedule before using this operation.")

    def add_activity(self, activity: Activity) -> None:
        """Add a single activity; fail when the ID already exists."""

        if activity.activity_id in self.activities:
            raise DuplicateActivityError(f"Duplicated ID: {activity.activity_id}")
        self.activities[activity.activity_id] = activity

    def add_activities(self, activities: Iterable[Activity]) -> None:
        """Add several activities at once using ``add_activity``."""

        for activity in activities:
            self.add_activity(activity)

    def activities_as_list(self) -> List[Activity]:
        """Return a stable list of activities for serialization or rebuilding.

        Use this helper when you need to persist or transmit the schedule
        state. The returned list is sorted by ``activity_id`` so you can feed
        it back into ``add_activities`` on a fresh ``ProjectSchedule`` to
        reconstruct the same plan.
        """

        return [self.activities[aid] for aid in sorted(self.activities)]

    def _validate_dependencies(self) -> None:
        """Ensure every predecessor refers to a known activity."""

        missing: Dict[str, List[str]] = defaultdict(list)
        for activity in self.activities.values():
            for pred in activity.predecessors:
                if pred not in self.activities:
                    missing[activity.activity_id].append(pred)
        if missing:
            lines = [f"{aid} -> {', '.join(sorted(preds))}" for aid, preds in sorted(missing.items())]
            raise UnknownPredecessorError("Missing dependencies: " + "; ".join(lines))

    def _topological_order(self) -> List[str]:
        """Return activity IDs in topological order for sequential calculations."""

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
            raise CycleError("Could not order activities because a cycle was detected.")

        return order

    def update_schedule(
        self,
        plot: bool = False,
        title: str = "Schedule",
        *,
        plot_resources: bool = False,
        resource_title: str = "Resource Histogram",
    ) -> Optional[Tuple[object, ...]]:
        """Compute dates, floats, and critical path; optionally return plots.

        Run this method after any activity/dependency change. The algorithm
        performs forward/backward passes to derive start/finish, total float,
        and critical path flags. When ``plot=True`` and/or ``plot_resources=True``
        it returns Gantt and resource histogram figures.
        """

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
                raise ScheduleError("early_start was not calculated for the activity.")

            total_float = (activity.late_start - activity.early_start).total_seconds() / 86400
            activity.total_float = total_float
            activity.is_critical = abs(total_float) < 1e-9

            if activity.delay > total_float:
                warnings.warn(
                    (
                        f"Activity {activity.activity_id} delay ({activity.delay} days) "
                        f"exceeds calculated total float ({total_float} days)."
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
            return (gantt, histogram) if histogram is not None else (gantt,)

        return None

    def plot_gantt(self, *, title: str = "Schedule") -> Tuple[Figure, Axes]:
        """Render a Gantt chart after ``update_schedule`` for quick visualization."""
        if not self.activities:
            raise ScheduleError("No activities available to generate the chart.")

        if any(activity.start is None or activity.finish is None for activity in self.activities.values()):
            raise ScheduleError("Run update_schedule before plotting the chart.")

        sorted_items: Sequence[Activity] = sorted(
            self.activities.values(), key=lambda a: a.start or self.start_date
        )

        fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_items) * 0.6)))
        yticks: List[int] = []
        ylabels: List[str] = []

        # Draw from top to bottom explicitly so the earliest activities appear at the top
        # of the chart without relying on axis inversion.
        positions: List[int] = list(range(len(sorted_items) - 1, -1, -1))

        for index, activity in zip(positions, sorted_items):
            start = activity.start or self.start_date
            duration: timedelta = (activity.finish or start) - start
            width_days: float = duration.days + duration.seconds / 86400
            ax.barh(index, width_days, left=start, align="center")

            midpoint = start + timedelta(days=width_days / 2)
            ax.text(midpoint, index, activity.name, va="center", ha="center")
            yticks.append(index)
            ylabels.append(f"{activity.activity_id} ({activity.area})")

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Date")
        ax.set_title(title)
        plt.tight_layout()
        return fig, ax

    def _build_resource_histogram(self) -> List[Dict[str, object]]:
        """Aggregate daily resource usage for later plotting."""
        if not self.activities:
            return []

        usage: Dict[datetime, Dict[str, float]] = defaultdict(
            lambda: {name: 0.0 for name in self.resource_names}
        )

        for activity in self.activities.values():
            if activity.start is None or activity.finish is None:
                raise ScheduleError("Run update_schedule before generating the resource histogram.")

            unknown = set(activity.resources or {}).difference(self.resource_names)
            if unknown:
                raise ScheduleError(
                    f"Activity {activity.activity_id} references unknown resources: {', '.join(sorted(unknown))}."
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
        title: str = "Resource Histogram",
        resources: Optional[Sequence[str]] = None,
    ) -> Tuple[Figure, List[Axes]]:
        """Plot one histogram per configured resource.

        Usage guidance:
        - After calling ``update_schedule``, invoke ``plot_resource_histogram``
          to produce a separate chart for each configured resource.
        - Use ``resources=["civil_engineers", "electrical_engineers"]`` to
          filter and render only selected resources. When ``resources`` is
          ``None``, all resources defined in ``ProjectSchedule.resource_names``
          are displayed.
        - Each subplot shows daily workload bars for its resource.
        """

        if not self.resource_histogram:
            raise ScheduleError("No histogram available. Run update_schedule first.")

        selected = list(resources) if resources is not None else list(self.resource_names)
        unknown = set(selected).difference(self.resource_names)
        if unknown:
            raise ScheduleError(
                f"Resources not configured for plotting: {', '.join(sorted(unknown))}."
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
            ax.set_ylabel("Load")
            ax.legend()

        axes[0].set_title(title)
        axes[-1].set_xlabel("Date")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig, axes

    def _progress_timeline(self) -> List[datetime]:
        """Build an inclusive daily timeline covering planned and actual data."""

        self._ensure_schedule_ready()

        start = min(activity.start or self.start_date for activity in self.activities.values())
        end = max(activity.finish or self.start_date for activity in self.activities.values())

        if self.progress_as_of is not None:
            start = min(start, self.progress_as_of)
            end = max(end, self.progress_as_of)

        days = (end.date() - start.date()).days
        return [datetime.combine(start.date(), time.min) + timedelta(days=offset) for offset in range(days + 1)]

    def _build_planned_progress_curve(self) -> List[Dict[str, object]]:
        """Compute the cumulative planned weight/percent over the timeline."""

        timeline = self._progress_timeline()
        total_weight = self._total_weight()
        if total_weight <= 0:
            raise ScheduleError("At least one activity with positive weight is required for S-curves.")

        curve: List[Dict[str, object]] = []
        for current in timeline:
            planned_weight = 0.0
            for activity in self.activities.values():
                if activity.start is None or activity.finish is None:
                    raise ScheduleError("Run update_schedule before generating S-curves.")

                duration_seconds = (activity.finish - activity.start).total_seconds()
                if duration_seconds <= 0:
                    fraction = 1.0 if current >= activity.finish else 0.0
                elif current <= activity.start:
                    fraction = 0.0
                elif current >= activity.finish:
                    fraction = 1.0
                else:
                    elapsed_seconds = (current - activity.start).total_seconds()
                    fraction = elapsed_seconds / duration_seconds

                planned_weight += activity.weight * fraction

            planned_percent = (planned_weight / total_weight) * 100
            curve.append({
                "date": current,
                "planned_weight": planned_weight,
                "planned_percent": planned_percent,
            })

        return curve

    def _build_actual_progress_curve(self, timeline: List[datetime]) -> List[Dict[str, object]]:
        """Compute the cumulative actual weight/percent using reported progress."""

        total_weight = self._total_weight()
        if total_weight <= 0:
            raise ScheduleError("At least one activity with positive weight is required for S-curves.")

        if self.progress_as_of is None:
            if any(a.progress_percent > 0 for a in self.activities.values()):
                raise ScheduleError("progress_as_of on the schedule is required when progress is reported.")
            return []

        cutoff = self.progress_as_of
        timeline = [point for point in timeline if point <= cutoff]
        if not timeline:
            return []

        curve: List[Dict[str, object]] = []
        for current in timeline:
            actual_weight = 0.0
            for activity in self.activities.values():
                actual_weight += activity.weight * (activity.progress_percent / 100)

            actual_percent = (actual_weight / total_weight) * 100
            curve.append({
                "date": current,
                "actual_weight": actual_weight,
                "actual_percent": actual_percent,
            })

        return curve

    def s_curve_data(self) -> Dict[str, List[Dict[str, object]]]:
        """Return planned and actual curves for progress comparisons.

        The actual curve is computed only up to ``progress_as_of`` on the
        schedule. When progress has been reported (any ``progress_percent``
        greater than zero), ``progress_as_of`` must be provided.
        """

        planned_curve = self._build_planned_progress_curve()
        timeline = [entry["date"] for entry in planned_curve]
        actual_curve = self._build_actual_progress_curve(timeline)
        return {"planned": planned_curve, "actual": actual_curve}

    def plot_s_curve(self, *, title: str = "S-Curve (Planned vs Actual)") -> Tuple[Figure, Axes]:
        """Plot the planned and actual S-curve using cumulative percent complete.

        The actual line is drawn only through the ``progress_as_of`` date of
        the schedule. When no progress is reported, the planned curve is still
        produced and the plot notes that actual data is absent.
        """

        curves = self.s_curve_data()
        planned = curves["planned"]
        actual = curves["actual"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot([p["date"] for p in planned], [p["planned_percent"] for p in planned], label="Planned", linewidth=2)

        if actual:
            ax.plot([a["date"] for a in actual], [a["actual_percent"] for a in actual], label="Actual", linewidth=2)
        else:
            ax.text(0.5, 0.1, "No actual progress reported", transform=ax.transAxes, ha="center", va="center")

        ax.set_ylabel("Cumulative percent")
        ax.set_xlabel("Date")
        ax.set_title(title)
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig, ax

    def get_activity(self, activity_id: str) -> Activity:
        """Return the activity by ID; raises KeyError when missing."""

        return self.activities[activity_id]

    def find_by_activity_id(self, activity_id: str) -> Activity:
        """Explicit alias of ``get_activity`` for filters or searches."""

        return self.get_activity(activity_id)

    def find_by_name(self, name: str) -> List[Activity]:
        """List activities whose name matches exactly (case-insensitive)."""

        return [a for a in self.activities.values() if a.name.lower() == name.lower()]

    def find_by_area(self, area: str) -> List[Activity]:
        """Return activities from a specific area (case-insensitive)."""

        return [a for a in self.activities.values() if a.area.lower() == area.lower()]

    def activities_on_date(self, target_date: date) -> List[Activity]:
        """List activities that take place (even partially) on a given day."""

        if any(a.start is None or a.finish is None for a in self.activities.values()):
            raise ScheduleError("Run update_schedule before querying by dates.")

        day_start = datetime.combine(target_date, time.min)
        day_end = day_start + timedelta(days=1)
        return [
            a
            for a in self.activities.values()
            if a.start < day_end and a.finish > day_start  # type: ignore[operator]
        ]

    def activities_in_period(self, start: date, end: date) -> List[Activity]:
        """List activities that intersect the period [start, end]."""

        if end < start:
            raise ValueError("The end date must be on or after the start date.")

        if any(a.start is None or a.finish is None for a in self.activities.values()):
            raise ScheduleError("Run update_schedule before querying by dates.")

        start_dt = datetime.combine(start, time.min)
        end_dt = datetime.combine(end, time.min) + timedelta(days=1)
        return [
            a
            for a in self.activities.values()
            if a.start < end_dt and a.finish > start_dt  # type: ignore[operator]
        ]

    def reset(self) -> None:
        """Clear schedules and activities, starting from scratch."""
        for activity in self.activities.values():
            activity.clear_schedule()
        self.activities.clear()
