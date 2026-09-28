---
doc_id: FS-04
title: Medical Provider Phantom Billing and Upcoding
line_of_business: health
synthetic: true
---
# FS-04 Medical Provider Phantom Billing and Upcoding

## Overview
A provider bills services that were never rendered (phantom billing), bills on dates the facility was closed,
or upcodes routine services to higher-reimbursement codes. Patients are often unaware of what was billed.

## Red-Flag Indicators
- RF-11 Billing anomaly: services billed on dates the clinic was closed, or upcoded line items.
- RF-07 Provider linked to several prior SIU-flagged claims.
- Claimant statement shows limited knowledge of treatment ("did not check the bills").

## Correlation Pattern
Billing anomalies (RF-11) at a provider that already appears in a flagged-claim cluster (RF-07) indicate
provider-driven fraud at HIGH likelihood. The claimant is usually a witness rather than a suspect.

## Investigation Steps
1. Reconcile billed dates against clinic opening records and appointment logs.
2. Run a coding audit over the provider's claims for the last 12 months.
3. Interview the claimant as a witness about the services actually received.

## Recommended Action
Refer to SIU (HIGH) with a provider-focused referral. Human sign-off required; do not penalise the claimant
without evidence of complicity.
