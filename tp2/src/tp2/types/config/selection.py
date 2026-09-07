import enum

from tp2.algorithms.selection.universal import universal_selection
from tp2.interfaces.algorithms.selection import Selection

from tp2.algorithms.selection.boltzmann import boltzmann_selection
from tp2.algorithms.selection.deterministic_tournament import deterministic_tournament_selection
from tp2.algorithms.selection.elite import elite_selection
from tp2.algorithms.selection.probabilistic_tournament import probabilistic_tournament_selection
from tp2.algorithms.selection.ranking import ranking_selection
from tp2.algorithms.selection.roulette import roulette_selection

class SelectionAlgorithms(enum.Enum):
    ELITE = "elite"
    ROULETTE = "roulette"
    UNIVERSAL = "universal"
    BOLTZMANN = "boltzmann"
    DETERMINISTIC_TOURNAMENT = "deterministic_tournament"
    STOCHASTIC_TOURNAMENT = "stochastic_tournament"
    RANKING = "rank"

    @property
    def function(self) -> Selection:
        functions: dict[SelectionAlgorithms, Selection] = {
            SelectionAlgorithms.ELITE: elite_selection,
            SelectionAlgorithms.ROULETTE: roulette_selection,
            SelectionAlgorithms.UNIVERSAL: universal_selection,
            SelectionAlgorithms.BOLTZMANN: boltzmann_selection,
            SelectionAlgorithms.DETERMINISTIC_TOURNAMENT: deterministic_tournament_selection,
            SelectionAlgorithms.STOCHASTIC_TOURNAMENT: probabilistic_tournament_selection,
            SelectionAlgorithms.RANKING: ranking_selection,
        }

        return functions[self]
