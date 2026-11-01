---
id: redis-runbook
title: Redis failover runbook
groups: eng
level: 2
---
# Redis failover runbook

The cache layer runs on Azure Cache for Redis with one primary and two replicas
in the same region. The application keeps working when the cache is unavailable:
every read falls back to the primary database, with a latency penalty that the
dashboards surface as a warning.

## Symptoms

A failover is in progress when the application logs show connection resets, the
cache hit ratio drops below forty percent and the synchronous publisher reports
that it lost the broadcast channel. A single replica disconnecting is not a
failover. Two replicas disconnecting within a minute is.

## Procedure

Confirm the primary is unreachable from two different app instances before
acting, so a single bad pod cannot trigger a cluster-wide response. Then request
a planned failover in the portal and wait for the new primary to be promoted. Do
not restart the application while promotion is in flight: the client reconnects
on its own and a restart only adds cold starts to an incident that is already
degraded.

## Credential rotation

Rotate the access key during a maintenance window, never in the middle of an
incident. Update the key in the platform secret store first, then restart one
instance, confirm it reconnects, and only then roll the rest.

## After the incident

Record the duration, the peak latency and whether the database fallback held.
Any failover longer than fifteen minutes gets a written postmortem.
