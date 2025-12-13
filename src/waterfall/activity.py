from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, List, Optional


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
    predecessors:
        Lista de IDs de atividades que devem terminar antes desta iniciar.
    start:
        Data de início calculada após o planejamento.
    finish:
        Data de término calculada após o planejamento.
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
    predecessors: List[str] = field(default_factory=list)
    start: Optional[datetime] = None
    finish: Optional[datetime] = None

    def set_schedule(self, start: datetime) -> None:
        """Ajusta datas de início e fim com base na duração configurada."""

        self.start = start
        self.finish = start + timedelta(days=self.duration)

    def depends_on(self, candidates: Iterable[str]) -> bool:
        """Retorna se a atividade depende de pelo menos um ID da lista."""

        predecessors = set(self.predecessors)
        return any(candidate in predecessors for candidate in candidates)

    def clear_schedule(self) -> None:
        """Remove informações de datas calculadas."""

        self.start = None
        self.finish = None
