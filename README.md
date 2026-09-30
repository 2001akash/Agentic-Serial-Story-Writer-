# Agentic Serial Story Writer

A resumable CLI for planning and writing a long serial with human approval at both the arc and episode level. It uses Groq or another OpenAI-compatible chat-completions endpoint; tests use a fake provider and need no API key.

## Quick Start (Windows)

Requires Python 3.11 or newer. Install the project in your virtual environment and create a local environment file:

```powershell
python -m pip install -e .
Copy-Item .env.example .env
notepad .env
```

Put your key only in the local `.env` file. It is git-ignored. Keep the matching model settings if they work for your Groq account, then run:

```powershell
python -m serial_writer create --id rider --premise "A delivery rider realizes every address on today's route belongs to someone who died in the same building."
python -m serial_writer plan rider
```

For unattended plan approval after all 200 outlines pass structural validation, add `--auto-approve` to the `plan` command. Manual plan approval remains the default; episode approval remains human-controlled.

Review and edit `.story-runs/rider/plan.json`, then approve it:

```powershell
python -m serial_writer approve-plan rider
python -m serial_writer draft rider
```

The draft appears in `.story-runs/rider/episodes/`. Edit that file directly if needed, then approve the edited text and leave persistent guidance for future episodes:

```powershell
python -m serial_writer approve rider --text-file .story-runs/rider/episodes/episode-001-draft.md --feedback "Keep Nia wary of the building manager."
```

Repeat `draft` and `approve` episode by episode. A run can be stopped at any point; `resume` shows the next unapproved episode, and `draft` continues from there:

```powershell
python -m serial_writer resume rider
python -m serial_writer draft rider
```

To rewrite an approved episode retroactively, edit a text file containing only the episode body and run:

```powershell
python -m serial_writer rewrite rider --episode 1 --text-file revised-episode-001.txt --feedback "The route changed before the first delivery."
```

That rewrite is reconciled against canon, later approved episodes are retained as `stale` history, and drafting resumes from the earliest stale episode. Commands accept `--runs-dir` to store runs somewhere other than `.story-runs`.

## Operations

- `python -m serial_writer status rider` shows plan status, episode counts, next episode, tokens, and cost.
- `python -m serial_writer events rider` prints timestamped model-call, retry, approval, rewrite, and failure records.
- `python -m serial_writer revise rider --feedback "Make the confrontation less predictable."` regenerates the pending draft.
- `python -m serial_writer estimate` estimates all 200 episodes; `--revision-rate`, `--seconds-per-call`, and `--episodes` adjust its assumptions.
- `python -m unittest discover -s tests -v` runs offline tests.

Set `STORY_PROVIDER` to `groq` or `openai`. Groq uses `GROQ_API_KEY` and defaults to `https://api.groq.com/openai/v1`; OpenAI uses `OPENAI_API_KEY` and defaults to `https://api.openai.com/v1`. `STORY_MODEL` selects the model, and `STORY_BASE_URL` can override the endpoint. Values in `.env` are loaded when the provider starts; already-set shell variables take precedence. Check Groq's current model catalog before running, since model availability and free-tier quotas can change. Groq has offered free developer usage subject to rate limits; it is not unlimited capacity, and this application cannot bypass provider quotas. `STORY_INPUT_USD_PER_MILLION` and `STORY_OUTPUT_USD_PER_MILLION` set the rates used for budgets and estimates. The defaults are illustrative `gpt-4o-mini` rates, not a live quote; set them to your selected model's rates for meaningful cost reporting.

## Design

`plan.json` is the editable 200-episode arc. Planning first creates a series blueprint, then requests outlines in batches of five. Partial valid responses are checkpointed and only missing episode numbers are requested again. Each completed batch is checkpointed in `run.json`; rerun `plan` after an interruption to continue without repeating completed outlines. Each run persists to `run.json`; approved episode text is also written as Markdown. Only approved episodes update canon. Drafting context is layered: the current episode outline and act, the rolling summary, a character/fact/open-thread ledger, the five most recent episode summaries, upcoming milestones, and accumulated editor guidance. Full prior prose is not repeatedly sent to the model.

The writer returns prose plus a continuity delta and a self-review in structured JSON. Up to two revision passes are allowed when the self-review flags a problem or the prose misses 400-700 words; the revisions explicitly target about 500 words. The result still waits for human approval. Approval feedback becomes durable guidance for later episodes. Editing an approved episode triggers a continuity extraction and marks dependent later episodes stale instead of silently treating them as valid.

Calls have a token ceiling, one transient retry, a per-episode cost cap, and a run budget. Calls, attempts, latency, token counts, estimated cost, failures, approvals, and rewrites are recorded. A response that would cross a budget is not accepted into the story state.

## Cost and Time

Run `python -m serial_writer estimate` for a calculation using the configured rates. The default estimate assumes 4,500 input and 1,250 output tokens per episode, a 25% revision rate, five-episode planning batches, and 18 seconds per model call. Human review time is additional. Actual totals depend heavily on the selected model, prompt size, provider pricing, retries, and rewritten episodes; inspect run telemetry and set a budget before generating.

To reduce spend, use a low-cost model for outline and continuity extraction, reserve a stronger model for episode prose, keep revisions exceptional, and compress the canonical ledger when it grows. The current implementation uses one configured model for all calls and does not automatically prune hard facts.

## Known Limits

This is a useful, auditable writing loop, not a guarantee of literary quality or perfect continuity. Canon extraction is model-generated and can omit implications; the rolling summary can drift; self-review is not an independent judge; and a human must review the plan, drafts, and any retroactive rewrite. The local JSON state is designed for one writer at a time, not concurrent processes or multi-user access. A live 200-episode demonstration, generated episode bundle, and screen recording require the evaluator's premise and model credentials; none are bundled as fabricated output.
## Demo

Screen recording: https://www.loom.com/share/130a3cbf1f134820916111e076726c1e

