# Biblioteca de planejamento waterfall

Este repositório agora entrega **um único arquivo (`waterfall.py`)** que pode ser copiado direto para a pasta do seu projeto.
Não há empacotamento com `pyproject.toml`, evitando conflitos com `uv` ou outros gerenciadores. Basta garantir que o arquivo
esteja no mesmo diretório dos seus scripts/notebooks (ou que ele esteja em um diretório presente no `PYTHONPATH`).

As principais entidades expostas são:

- `Activity`: descreve uma atividade com informações de duração, recursos (customizáveis), dependências e atraso (delay) opcional.
- `ProjectSchedule`: mantém a lista de atividades, calcula datas de início e fim a partir das precedências, deriva early/late start,
  folga e caminho crítico, além de produzir gráficos de Gantt e histogramas de recursos nomeados.

> Dependência única: `matplotlib` (instale manualmente com `pip install matplotlib` ou `uv pip install matplotlib`).

## Uso básico

```python
from datetime import datetime
import waterfall as wf

projeto = wf.ProjectSchedule(
    start_date=datetime(2025, 1, 6),
    resource_names=["pedreiros", "eletricistas", "escavadeiras"],  # nomes e quantidade de recursos configuráveis
)

atividades = [
    wf.Activity(
        name="Levantamento",
        activity_id="A1",
        area="Análise",
        short_description="Entrevistas",
        long_description="Entrevistas com stakeholders",
        duration=5,
        resources={"pedreiros": 1},
    ),
    wf.Activity(
        name="Modelagem",
        activity_id="A2",
        area="Arquitetura",
        short_description="Diagramas",
        long_description="Modelagem UML",
        duration=3,
        resources={"eletricistas": 1},
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

### Histograma de recursos (nomes e quantidades flexíveis)

- Defina a quantidade e o nome dos recursos no `ProjectSchedule` (`resource_names=["pedreiros", "eletricistas"]` ou apenas `resource_count=5`).
- Cada `Activity` pode receber cargas via o dicionário `resources={"pedreiros": 2, "eletricistas": 1}`.
- Após `update_schedule`, gere um histograma separado para cada recurso (ou só para os desejados):

```python
# Todos os recursos configurados
fig, axes = projeto.plot_resource_histogram(title="Recursos do projeto")

# Somente pedreiros
fig, axes = projeto.plot_resource_histogram(resources=["pedreiros"], title="Pedreiros por dia")
```

### Consultas rápidas

- `find_by_name("Modelagem")`, `find_by_activity_id("A1")`, `find_by_area("Arquitetura")`
- `activities_on_date(date(2025, 1, 8))` retorna atividades que tocam aquele dia.
- `activities_in_period(date(2025, 1, 6), date(2025, 1, 10))` traz atividades que intersectam o intervalo.

## Como usar sem instalar nada

1. Copie o arquivo `waterfall.py` para a pasta do seu projeto (onde seu script ou notebook está salvo).
2. Garanta que o `matplotlib` esteja instalado (`pip install matplotlib` ou `uv pip install matplotlib`).
3. Importe normalmente: `import waterfall as wf`.

Se preferir não copiar, também funciona manter o arquivo em outro diretório e incluir esse caminho no `PYTHONPATH` antes de rodar
seus scripts/notebooks.
