# Biblioteca de planejamento waterfall

Esta biblioteca fornece classes para planejar projetos no modelo waterfall. As principais entidades são:

- `Activity`: descreve uma atividade com informações de duração, recursos, dependências e atraso (delay) opcional.
- `ProjectSchedule`: mantém a lista de atividades, calcula datas de início e fim a partir das precedências, deriva early/late start,
  folga e caminho crítico, além de produzir gráficos de Gantt e histogramas de recursos.

## Uso básico

```python
from datetime import datetime
import waterfall as wf

projeto = wf.ProjectSchedule(start_date=datetime(2025, 1, 6))

atividades = [
    wf.Activity(
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
    wf.Activity(
        name="Modelagem",
        activity_id="A2",
        area="Arquitetura",
        short_description="Diagramas",
        long_description="Modelagem UML",
        duration=3,
        resource1=0,
        resource2=1,
        resource3=0,
        delay=1.5,
        predecessors=["A1"],
    ),
]

projeto.add_activities(atividades)
projeto.update_schedule(plot=True, plot_resources=True)

for atividade in atividades:
    print(
        atividade.activity_id,
        "ES=", atividade.early_start,
        "LS=", atividade.late_start,
        "float=", atividade.total_float,
        "critica=", atividade.is_critical,
        "delay=", atividade.delay,
    )
```

O método `update_schedule` calcula as datas de início e fim com base nas dependências, a folga (diferença entre late e early start)
e marca `is_critical` quando a folga é zero. Se o `delay` configurado em uma atividade exceder a folga calculada,
um *warning* é emitido, mas o cronograma é ajustado com o atraso solicitado. Também pode devolver um gráfico de Gantt quando `plot=True`.

### Histograma de recursos

Ao rodar `update_schedule`, o cronograma também calcula um histograma diário somando os recursos alocados.
Para visualizar:

```python
fig, ax = projeto.plot_resource_histogram()
```

O método `plot_resource_histogram` plota barras empilhadas por data para os três recursos cadastrados.
