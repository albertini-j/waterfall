# Biblioteca de planejamento waterfall

Esta biblioteca fornece classes para planejar projetos no modelo waterfall. As principais entidades são:

- `Activity`: descreve uma atividade com informações de duração, recursos e dependências.
- `ProjectSchedule`: mantém a lista de atividades, calcula datas de início e fim a partir das precedências e produz gráficos de Gantt.

## Uso básico

```python
from datetime import datetime
from waterfall import Activity, ProjectSchedule

projeto = ProjectSchedule(start_date=datetime(2025, 1, 6))

atividades = [
    Activity(
        name="Levantamento",
        activity_id="A1",
        area="Análise",
        short_description="Entrevistas",
        long_description="Entrevistas com stakeholders",
        duration=5,
        resource1=1,
        resource2=0,
        resource3=0,
    ),
    Activity(
        name="Modelagem",
        activity_id="A2",
        area="Arquitetura",
        short_description="Diagramas",
        long_description="Modelagem UML",
        duration=3,
        resource1=0,
        resource2=1,
        resource3=0,
        predecessors=["A1"],
    ),
]

projeto.add_activities(atividades)
projeto.update_schedule(plot=True)
```

O método `update_schedule` calcula as datas de início e fim com base nas dependências e pode devolver um gráfico de Gantt quando `plot=True`.
