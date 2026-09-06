---
status: accepted
---

# A failed Completion Audit continues the current Run

Implementation status: implemented as the explicit
`CompletionPolicy(CompletionMode.ENFORCE, max_retries=2)` option. The default
`OBSERVE` mode preserves compatibility for applications using evaluation as
feedback. Enforcement requires a bound plan and an available monitor; it applies
to both text completion and a Tool's `COMPLETE` control.

Under enforcement, a Completion Claim is not terminal until every Goal Requirement and Constraint
has scope-appropriate Evidence. When that audit fails and the Run still has
budget, the evaluator's unmet items and missing Evidence are projected at the
next Decision Boundary in the same Run. This preserves the causal window and
keeps evaluator feedback distinct from a new user task. A new Run is reserved
for an explicit follow-up or retry after the current Run has otherwise become
terminal.

The rejected alternative was to commit the actor's Completion Claim and start
a corrective Run. That would make an unverified claim durable Conversation
truth, split one causal trajectory across Runs, and require rollback semantics
that the current Session model does not provide.


## Retry bounds and uncertainty

A false, unknown, or conflicting verdict does not approve completion. Temporary
evidence failures may recover at a later checkpoint. Invalid monitor protocols
stop the gated Run because further capture is disabled. Additional completion
attempts are bounded independently, while Actor turn/token limits and cancellation
remain effective. Exhaustion fails with `completion_audit_failed` and does not
advance Conversation revision. Rejected final prose is retained in Audit only;
completed tool observations remain available without breaking call/result ordering.

## Why enforcement is explicit

Unknown evidence and judge outages can block otherwise useful work, while an LLM
can still misjudge semantic quality despite valid citations. Observation mode
lets hosts measure those cases before making approval mandatory. This choice is
based on concrete regression cases in `tests/test_semantic_evaluation.py`: invalid
judge output, missing usage, timeouts, budget exhaustion, same-Run recovery,
rejected-claim exclusion, and bounded retries. These tests establish mechanics;
they do not establish a production judge's false-positive or false-negative rate.
Applications must calibrate their criteria against representative tasks.
