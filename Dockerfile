# Slim submission image (~0.5 GB vs 7.3 GB for the baselines image).
# The evaluation cluster gives the pod only 20 minutes to reach Running,
# which includes pulling this image — the full flatland-baselines image
# (conda + flatland-rl[ml] with torch/ray + ffmpeg) timed that out.
#
# Contract with the evaluator (see flatland-benchmarks orchestrator):
# the container is started with args
#   flatland-trajectory-generate-from-policy --data-dir ... --env-path ... --ep-id ...
# so the CLI must be on PATH, POLICY/OBS_BUILDER env vars set, and our
# code importable. No ENTRYPOINT: args run directly.
FROM python:3.12-slim

WORKDIR /app

COPY submission/requirements.txt submission/requirements.txt
RUN pip install --no-cache-dir -r submission/requirements.txt

COPY submission/ submission/
COPY reinforcement_learning/ reinforcement_learning/

ENV POLICY=submission.my_policy.MyPolicy
ENV OBS_BUILDER=submission.my_observation_builder.MyObservationBuilder
ENV PYTHONPATH=/app
