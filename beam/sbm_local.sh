#!/bin/bash
#
# Utilize docker-compose to run beam-pipeline locally in the same environment
# as the remote workers.
#


OUTPUT_PATH="/tmp/sbm"

rm -rf "${OUTPUT_PATH}"
mkdir -p ${OUTPUT_PATH}

docker-compose run \
  --entrypoint "python3 /app/beam_sbm.py \
  --output ${OUTPUT_PATH} \
  --nsamples 5 \
  --nvertex_min 10 --nvertex_max 15 \
  --nedges_min 5 --nedges_max 10 \
  --runner=DirectRunner" \
  research-graph-synthetic 



