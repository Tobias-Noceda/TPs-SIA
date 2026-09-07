import enum

from tp2.interfaces.algorithms.selection import Selection

from tp2.algorithms.selection.elite import elite_selection

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
            SelectionAlgorithms.ROULETTE: elite_selection,
            SelectionAlgorithms.UNIVERSAL: elite_selection,
            SelectionAlgorithms.BOLTZMANN: elite_selection,
            SelectionAlgorithms.DETERMINISTIC_TOURNAMENT: elite_selection,
            SelectionAlgorithms.STOCHASTIC_TOURNAMENT: elite_selection,
            SelectionAlgorithms.RANKING: elite_selection,
        }

        return functions[self]