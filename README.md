# AI Companion Eval Agents

This repository contains my submission work for an automated evaluation task on AI companion apps. It has two main parts:

- `Q1`: large-scale dataset construction from app-store metadata plus browser-and-LLM evaluation
- `Q2`: platform automation and reverse-engineering experiments, starting from Yollo and then extending toward cross-site API probing

## Repository Layout

### Q1 results
- `outputs/evaluated_apps_submission.csv`
  - final cleaned Q1 submission-style result file

### Core Q1 files

- `build_input_apps_csv.py`
  - builds `input_apps.csv` from `app_store_apps_details.json` and `google_play_apps_details.json`
- `q1_agent_pipeline.py`
  - main browser + LLM evaluation pipeline
- `repair_outputs.py`
  - post-processing / cleanup for Q1 outputs
- `merge_human_verify.py`
  - merges manually reviewed unresolved cases
- `outputs/unresolved_urls.csv`
  - unresolved cases retained after review

### Core Q2 files

- `q2_yollo_poc.py`
  - hand-built Yollo API proof of concept
- `q2_yollo_api_probe_agent.py`
  - closed-loop Yollo network probe using Playwright + LLM + HTTP verification
- `q2_yollo_probe_replay.py`
  - replay script that uses the probe-derived plan to run the same 10-turn conversation
- `q2_site_probe_agent_v2.py`
  - broader second-generation cross-site probe for other companion web apps

### Q2 result artifacts kept in the repo

- `yollo_api_probe_output/`
  - successful Yollo probe output
- `yollo_api_probe_replay_output/`
  - successful 10-turn replay from the derived plan
- `girlfriend_ai_probe_output_v2/`
- `charclub_probe_output_v2/`
- `dopple_probe_output_v2/`
- `bella_probe_output_v2/`
- `bubblechat_probe_output_v2/`
- `kavana_probe_output_v2/`
- `livec_probe_output_v2/`
- `trumate_probe_output_v2/`
  - representative cross-site probe attempts used to analyze failure modes and next steps

### Write-up

- `descript.tex`
- `descript.pdf`

## Environment

Create a `.env` file from `.env.example` and fill in your API key.

The code expects an OpenAI-compatible endpoint and currently defaults to:

- `OPENAI_BASE_URL=https://api.chatanywhere.org/v1`
- `OPENAI_MODEL=gpt-4.1-mini`

## Suggested Setup

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Typical Q1 Workflow

```powershell
python build_input_apps_csv.py
python q1_agent_pipeline.py
python repair_outputs.py
python merge_human_verify.py
```

## Typical Q2 Workflow

### Manual Yollo PoC

```powershell
python q2_yollo_poc.py
```

### Closed-loop probe on Yollo

```powershell
python q2_yollo_api_probe_agent.py
python q2_yollo_probe_replay.py
```

### Cross-site probe

```powershell
python q2_site_probe_agent_v2.py --target-url https://example.com --output-dir example_probe_output_v2
```

## Notes

- Some result directories are intentionally kept because they are discussed in the write-up and show both success and failure modes.
- `.env`, Python caches, LaTeX build intermediates, and other transient files are ignored or removed for cleaner version control.
