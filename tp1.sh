#!/bin/bash

TIMEOUT=300

maps=$(find ./tp1/res/boards -type f -name '*.json' -not -path "*/too_hard/*")

for map in $maps
do
    for algo in "a_star" "greedy"
    do
        for heur in "closest" "reduction"
        do
            timeout $TIMEOUT uv run tp1 "$algo" "$heur" "$map"
        done
    done

    for algo in "iddfs" "dfs" "bfs"
    do
        timeout $TIMEOUT uv run tp1 "$algo" "$map"
    done
done
