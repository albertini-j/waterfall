# Waterfall scheduling helper

This repository ships a **single module file (`waterfall.py`)**. Copy it into your project folder and import it directly (`import waterfall as wf`) without installing a package. External dependencies: `matplotlib` (for plotting) and `pydantic` (for typed validation). Install them manually with `pip install matplotlib pydantic` or `uv pip install matplotlib pydantic`.

Main entities:
- `Activity`: describes a task with duration, area, descriptions, named resources, predecessors, optional delay, **weight**, and **progress tracking** fields (`progress_percent`, `actual_finish`).
- `ProjectSchedule`: stores activities, computes start/finish dates from precedence, derives early/late start, total float, critical path, and produces Gantt charts, per-resource histograms, and planned vs. actual **S-curves**. It holds a single `progress_as_of` date that applies to all activities when drawing the actual curve. It is a Pydantic `BaseModel`, so field validation and serialization are available by default.

## Basic usage
```python
from datetime import datetime
import waterfall as wf

schedule = wf.ProjectSchedule(
    start_date=datetime(2025, 1, 6),
    resource_names=["civil_engineers", "electrical_engineers", "survey_specialists"],
    progress_as_of=datetime(2025, 1, 8),
)

activities = [
    wf.Activity(
        name="Field survey",
        activity_id="A1",
        area="Analysis",
        short_description="Stakeholder interviews",
        long_description="Interviews and onsite assessment",
        duration=5,
        weight=1.5,
        resources={"survey_specialists": 2},
    ),
    wf.Activity(
        name="Solution design",
        activity_id="A2",
        area="Architecture",
        short_description="Diagrams",
        long_description="Solution and integration diagrams",
        duration=3,
        weight=2.0,
        resources={"civil_engineers": 1},
        delay=1.5,
        predecessors=["A1"],
        progress_percent=50,
    ),
]

schedule.add_activities(activities)
schedule.update_schedule(plot=True, plot_resources=True)

# Optional: plot planned vs. actual S-curve using weights and reported progress
schedule.plot_s_curve(title="Progress S-curve")

for activity in activities:
    print(
        activity.activity_id,
        "ES=", activity.early_start,
        "LS=", activity.late_start,
        "float=", activity.total_float,
        "critical=", activity.is_critical,
        "delay=", activity.delay,
    )
```

`update_schedule` calculates start/finish dates based on dependencies, total float (late minus early start), and flags `is_critical` when float is zero. If an activity `delay` exceeds the calculated float, a warning is emitted while still applying the requested delay. Use `plot=True` for a Gantt chart and `plot_resources=True` for resource histograms.

### Resource histogram (flexible names and counts)
- Define the number and names of resources on `ProjectSchedule` (`resource_names=["civil_engineers", "electrical_engineers"]` or just `resource_count=5`).
- Each `Activity` can carry workloads via the dictionary `resources={"civil_engineers": 2, "electrical_engineers": 1}`.
- After `update_schedule`, generate one histogram per resource (or just the ones you want):

```python
# All configured resources
fig, axes = schedule.plot_resource_histogram(title="Resource load")

# Only electrical engineers
fig, axes = schedule.plot_resource_histogram(resources=["electrical_engineers"], title="Electrical engineers per day")
```

### Quick queries
- `find_by_name("Solution design")`, `find_by_activity_id("A1")`, `find_by_area("Architecture")`
- `activities_on_date(date(2025, 1, 8))` returns activities touching that day.
- `activities_in_period(date(2025, 1, 6), date(2025, 1, 10))` returns activities intersecting the interval.
- `activities_as_list()` returns a deterministic list of activities (sorted by ID) so you can serialize or rebuild a schedule elsewhere with `add_activities`.
- `s_curve_data()` returns planned and actual cumulative progress (weights and percents) ready for custom plotting; `plot_s_curve()` renders the chart directly.

### Progress tracking and S-curves
- Each activity accepts `weight` (positive float) to contribute to planned value.
- Report progress with `progress_percent` (0–100) and set a **schedule-level** `progress_as_of` date when any progress is reported.
- `actual_finish` can only be set when `progress_percent` reaches 100%.
- Call `plot_s_curve()` to visualize planned vs. actual progress using the weights and reported dates. The actual line stops at `progress_as_of`.

## Full example (5 activities using every function)
The example below shows the full workflow with five activities, including scheduling, plotting, resource histogram filtering, S-curve progress tracking, and query helpers.

```python
from datetime import datetime, date
import waterfall as wf

# 1) Configure schedule and named resources
schedule = wf.ProjectSchedule(
    start_date=datetime(2025, 3, 3),
    resource_names=["civil_engineers", "electrical_engineers", "geotechnical_specialists"],
    progress_as_of=datetime(2025, 3, 7),
)

# 2) Create activities (five total) with dependencies, resources, optional delays,
#    weights (for S-curve), and progress snapshots
activities = [
    wf.Activity(
        name="Feasibility study",
        activity_id="A1",
        area="Planning",
        short_description="Feasibility",
        long_description="Initial scope assessment",
        duration=2,
        weight=1.0,
        resources={"civil_engineers": 1},
        progress_percent=100,
        actual_finish=datetime(2025, 3, 4),
    ),
    wf.Activity(
        name="Detailed engineering",
        activity_id="A2",
        area="Engineering",
        short_description="Design",
        long_description="Technical detailing",
        duration=4,
        weight=2.0,
        resources={"electrical_engineers": 1},
        predecessors=["A1"],
        progress_percent=40,
    ),
    wf.Activity(
        name="Site preparation",
        activity_id="A3",
        area="Field",
        short_description="Preparation",
        long_description="Site setup and logistics",
        duration=3,
        weight=1.5,
        resources={"civil_engineers": 2, "geotechnical_specialists": 1},
        predecessors=["A1"],
    ),
    wf.Activity(
        name="Foundation",
        activity_id="A4",
        area="Field",
        short_description="Foundations",
        long_description="Foundation execution",
        duration=5,
        weight=3.0,
        resources={"civil_engineers": 3, "geotechnical_specialists": 1},
        predecessors=["A3"],
        delay=1,  # intentional lag
    ),
    wf.Activity(
        name="Electrical systems",
        activity_id="A5",
        area="Engineering",
        short_description="Electrical",
        long_description="Power and cabling infrastructure",
        duration=4,
        weight=2.5,
        resources={"electrical_engineers": 2},
        predecessors=["A2", "A4"],
    ),
]

# 3) Register activities (one by one with add_activity, or in bulk)
schedule.add_activities(activities)

# 4) Calculate schedule and generate plots (Gantt + per-resource histograms)
gantt_fig, resource_fig = schedule.update_schedule(
    plot=True,
    plot_resources=True,
    title="Complete schedule",
    resource_title="Daily resource usage",
)

# 5) Plot planned vs. actual S-curve (weights + reported progress)
progress_fig, progress_ax = schedule.plot_s_curve(title="Planned vs. actual progress")

# 6) Queries: by name, ID, area, and date ranges
print(schedule.find_by_name("Foundation"))
print(schedule.find_by_activity_id("A3"))
print(schedule.find_by_area("Field"))
print(schedule.activities_on_date(date(2025, 3, 6)))
print(schedule.activities_in_period(date(2025, 3, 3), date(2025, 3, 10)))

# 7) Filtered histograms (only geotechnical specialists, for example)
geotech_fig, geotech_axes = schedule.plot_resource_histogram(
    resources=["geotechnical_specialists"],
    title="Geotechnical specialists per day",
)

# 8) Reset everything and start over if needed
schedule.reset()
```

## Using without installation
1. Copy `waterfall.py` to your project folder (where your script or notebook lives).
2. Ensure `matplotlib` is installed (`pip install matplotlib` or `uv pip install matplotlib`).
3. Import normally: `import waterfall as wf`.

If you keep the file in another folder, add that path to `PYTHONPATH` before running your scripts/notebooks.
