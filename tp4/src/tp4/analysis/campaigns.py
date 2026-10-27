from typing import Dict, List

import numpy as np
import pandas as pd

# Creator selection for the agency's campaigns (exercise 1.4). Every score is the mean z-score over
# the whole catalog of the variables that measure an objective, so it reads as "standard deviations
# above the average creator" and doesn't depend on the variables' units.

# Client → topic column of its audience
TOPICS: Dict[str, str] = {
    "football": "topic_football_pct",
    "makeup": "topic_makeup_pct",
    "dogs": "topic_dogs_pct",
    "art": "topic_art_pct",
}

# Objective → variables that measure it (all of them "more is better")
OBJECTIVES: Dict[str, List[str]] = {
    # Massive reach: as many accounts as possible, in Argentina
    "reach": ["average_reach", "estimated_argentina_reach"],
    # Discovery: reaching people who don't follow the creator yet and getting it shared
    "discovery": ["non_follower_reach_pct", "shares_per_1000", "trend_posts_pct", "short_form_posts_pct"],
    # Educational depth: content that explains, is saved and watched for longer
    "depth": ["content_tutorial_pct", "saves_per_1000", "average_watch_time_seconds", "average_caption_words", "comments_per_1000"],
    # Conversion (enquiries, adoptions, sales): the audience acts after the post and is in Argentina
    "conversion": ["profile_visits_per_1000", "comments_per_1000", "saves_per_1000", "argentina_audience_pct"],
}

def dominant_topic(df: pd.DataFrame) -> pd.Series:
    """Client whose topic has the largest share of each creator's posts."""
    inverse = {column: client for client, column in TOPICS.items()}
    return df[list(TOPICS.values())].idxmax(axis=1).map(inverse)

def objective_scores(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Score of every row on every objective: mean z-score of its variables, standardized with the
    reference's mean and population std (default: df itself; e.g. the catalog to score map neurons).
    """
    reference = df if reference is None else reference
    scores = {}
    for objective, columns in OBJECTIVES.items():
        values, base = df[columns], reference[columns]
        scores[objective] = ((values - base.mean()) / base.std(ddof=0)).mean(axis=1)
    return pd.DataFrame(scores, index=df.index)

def targeted_reach(df: pd.DataFrame, client: str) -> pd.Series:
    """
    Estimated Argentine accounts reached that are interested in the client's topic, assuming the
    audience's interest matches the share of posts on that topic.
    """
    return df["estimated_argentina_reach"] * df[TOPICS[client]] / 100

def select(df: pd.DataFrame, client: str, objective: str, count: int, candidates: pd.Series | None = None) -> pd.DataFrame:
    """
    The count best creators for a client and objective: only creators whose dominant topic is the
    client's (or the given candidates mask), ranked by the objective score.
    """
    mask = dominant_topic(df) == client if candidates is None else candidates
    ranked = df[mask].assign(score=objective_scores(df)[objective][mask])
    return ranked.sort_values("score", ascending=False).head(count)

def select_budget(df: pd.DataFrame, client: str, objective: str, budget: float) -> pd.DataFrame:
    """Best creators for a client and objective, in ranking order, while one post from each fits the budget (USD)."""
    ranked = select(df, client, objective, len(df))
    return ranked[ranked["price_usd"].cumsum() <= budget]

def by_followers(df: pd.DataFrame, count: int) -> pd.DataFrame:
    """The agency's current strategy: the count creators with the most followers, whatever they post."""
    return df.sort_values("followers", ascending=False).head(count)

def compare(df: pd.DataFrame, selections: Dict[str, pd.DataFrame], client: str) -> pd.DataFrame:
    """Per selection: what the client gets for one sponsored post with each selected creator."""
    rows = {}
    for name, selection in selections.items():
        reach = targeted_reach(selection, client)
        cost = selection["price_usd"].sum()
        rows[name] = {
            "creators": ", ".join(selection["name"]),
            "on_topic": float((dominant_topic(selection) == client).mean()),
            "topic_pct": float(selection[TOPICS[client]].mean()),
            "argentina_audience_pct": float(selection["argentina_audience_pct"].mean()),
            "followers": float(selection["followers"].sum()),
            "reach": float(selection["average_reach"].sum()),
            "targeted_reach": float(reach.sum()),
            "non_follower_reach_pct": float(selection["non_follower_reach_pct"].mean()),
            "saves_per_1000": float(selection["saves_per_1000"].mean()),
            "profile_visits_per_1000": float(selection["profile_visits_per_1000"].mean()),
            "tutorial_pct": float(selection["content_tutorial_pct"].mean()),
            "cost_usd": float(cost),
            "usd_per_1000_targeted": float(1000 * cost / max(reach.sum(), 1e-12)),
            # Profile visits from the targeted audience: a proxy of enquiries / store visits
            "targeted_profile_visits": float((reach * selection["profile_visits_per_1000"] / 1000).sum()),
            "targeted_saves": float((reach * selection["saves_per_1000"] / 1000).sum()),
        }
    return pd.DataFrame(rows).T

def neighbors_on_map(winners: np.ndarray, grid_distances: np.ndarray, creator: int) -> np.ndarray:
    """Creators whose winner is the same neuron as creator's or adjacent to it: alternatives with a similar profile."""
    distance = grid_distances[winners[creator], winners]
    mask = distance <= 1 + 1e-9
    mask[creator] = False
    return np.flatnonzero(mask)
