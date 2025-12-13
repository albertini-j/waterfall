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
    delay:
        Atraso aplicado ao início da atividade (em dias). Não bloqueia o planejamento, mas pode reduzir a folga.
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
