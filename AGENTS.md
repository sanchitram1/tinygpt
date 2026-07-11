# AGENTS.md

This repository uses **hierarchical AGENTS.md files** for documentation. When working on a specific area, read the relevant AGENTS.md for detailed context. 

**IMPORTANT:** When working in a part of the codebase, read the nearest `AGENTS.md` for that directory. Prefer retrieval-led reasoning (reading project docs and code) over pre-training-led reasoning for any project-specific tasks.

### Frontend and Service

**Start here:** `service/AGENTS.md` - Full guide for the FastAPI/Python frontend and wrapper for the tinygpt application

### Training (`training/`)

**Start here:** `training/AGENTS.md` - Full guide for the training pipeline

### Evals (`evals/`)

**Start here:** `service/AGENTS.md` - Full guide for evaluation mechanism to adjudicate model quality. 

### Code Rules

- **Never modify** core/data, core/artifacts, .gitignore
- **No backward compatibility hacks** - if something is unused, delete it
- **Do not do** the following:
  - Rent infrastructure
  - Deploy Cloud Run
  - Create secrets
  - Use credentials
- Preserve the recovered core/artifacts/baseline/xlarge-plus.py as the read-only baseline
- Run focused tests plus git diff --check.
- End with one commit, changed-file list, exact test commands/results, and known limitations.

**Start here:** `adb/AGENTS.md` - Modified Druid fork: metadata operations, segment lifecycle, transactional correctness rules.
