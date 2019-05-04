#!/usr/bin/env bash

source ../venv/bin/activate

for i in 0,1 0,2 0,3 1,2 1,3 2,3; do IFS=","
	set -- $i;
	python3 evaluate.py --dataset iris --features $1 $2 --decision-boundary --no-plot --safe
done

deactivate