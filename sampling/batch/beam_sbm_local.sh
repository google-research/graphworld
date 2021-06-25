#!/bin/bash
#
# Utilize docker-compose to run beam-pipeline locally in the same environment
# as the remote workers.
#

docker-compose run \
  --entrypoint "python3 /app/beam_sbm.py --output /app/data/beam_sbm --nsamples 5 --runner=DirectRunner" \
  research-graph-synthetic 



