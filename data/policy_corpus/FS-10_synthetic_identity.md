---
doc_id: FS-10
title: Synthetic Identity and Policy-Stacking Fraud
line_of_business: multi-line
synthetic: true
---
# FS-10 Synthetic Identity and Policy Stacking

## Overview
Fabricated or blended identities purchase several policies and file early claims, reusing contact details across
supposedly unrelated claimants.

## Red-Flag Indicators
- RF-17 Phone or address shared with unrelated claimants.
- RF-01 Loss shortly after inception.
- RF-08 Many claims across several newly created policies.

## Correlation Pattern
Shared contact data (RF-17) across claimants with new policies (RF-01) indicates possible synthetic identity; this
often overlaps with staged-collision rings (see FS-01).

## Investigation Steps
1. Verify identity documents through the approved verification service.
2. Map all policies and claims sharing the contact data.

## Recommended Action
Refer to SIU when identity cannot be verified; otherwise monitor.
