---
doc_id: FS-08
title: Duplicate Claims / Double Dipping Across Carriers
line_of_business: multi-line
synthetic: true
---
# FS-08 Duplicate Claims Across Carriers

## Overview
The same loss is reported to more than one insurer (or twice to the same insurer) to be paid multiple times.

## Red-Flag Indicators
- RF-13 Cross-carrier claim-index match on the same claimant, loss date and loss type.
- Same repair invoice or medical bill submitted under more than one policy.

## Correlation Pattern
A confirmed index match (RF-13) on claimant, loss date and loss type is by itself a HIGH-likelihood duplicate-claim
signal, because legitimate other-insurance situations must be disclosed on the claim form.

## Investigation Steps
1. Contact the other carrier through the approved information-sharing channel to confirm payment.
2. Compare invoices and photos submitted to both carriers.
3. Check the claim form's other-insurance disclosure.

## Recommended Action
Refer to SIU (HIGH). Suspend further payment pending human SIU review; never auto-deny.
