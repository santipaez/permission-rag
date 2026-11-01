---
id: security-playbook
title: Security incident playbook
groups: security,eng
level: 2
---
# Security incident playbook

Any suspected identity compromise starts with containment, not investigation.
The first action is to revoke the sessions of the affected account, which
invalidates refresh tokens immediately. Disabling the account comes next if the
user cannot be reached within fifteen minutes.

## Severity

A confirmed credential leak on an account with administrative roles is severity
one. The same leak on a standard account is severity two. A phishing email that
nobody clicked is not an incident; it is a report.

## Evidence

Preserve the sign-in logs before changing anything else. Export the last seven
days of Entra ID sign-in records and the conditional access evaluation results
for the account. Doing this after remediation usually loses the detail that
explains how the attacker got in.

## Communication

The on-call engineer notifies the security lead within thirty minutes of a
severity one. Clients are informed only after the facts are confirmed, and the
message is written by the account owner, not by the engineer who found it.

## Review

Every severity one incident produces a review within five business days. The
review answers one question: which control would have stopped this, and why did
it not.
