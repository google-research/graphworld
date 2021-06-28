#!/bin/bash
#
# Utilize docker-compose to run beam-pipeline locally in the same environment
# as the remote workers.
#

# rm  ./src/data/beam_sbm/*

docker-compose run \
  --entrypoint "python3 /app/beam_sbm.py \
  --output /tmp/sbm \
  --nsamples 5 \
  --nvertex_min 10 --nvertex_max 15 \
  --nedges_min 5 --nedges_max 10 \
  --runner=DirectRunner" \
  research-graph-synthetic 



