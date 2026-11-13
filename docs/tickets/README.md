# Tickets

Records for the CPU decode work on `feature/cpu-decode`. Each folder has the
ticket definition, the design that was implemented, the milestones, the
verification, and the caveats. The files are separate so a later change can
update one of them without rewriting the others.

| Ticket | Subject | Code |
| --- | --- | --- |
| [QI-001](QI-001-llama-cpp-compare/definition.md) | llama.cpp same-machine comparison and `--threads` | on this branch, with QI-003 |
| [QI-002](QI-002-contributor-loop/definition.md) | `AGENTS.md` and the runbook corrections | on this branch |
| [QI-003](QI-003-output-column-split/definition.md) | Decode matmul split across output columns | on this branch, with QI-001 |
| [QI-004](QI-004-tensor-view/definition.md) | Shared F32 storage and offset tensor views | `6b8935a` |
| [QI-005](QI-005-mmap-load/definition.md) | Map GGUF and QIF files at load | `77db1f3` |

QI-006 and later tickets are not started. Their definitions stay in the
roadmap discussion and are not archived here until there is a design and a
verification record.
