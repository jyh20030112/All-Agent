---
status: accepted
---

# A failed Completion Audit continues the current Run

A Completion Claim is not terminal until every Goal Requirement and Constraint
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
