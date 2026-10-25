# Streamlit UI

GitHub Pages cannot run live Mercari search or an LLM. This app is a thin Streamlit wrapper around the same `LiveShopper` the CLI uses. Start it locally or with Docker. Secrets stay in the environment and are never sent to the browser.

## Required environment

Copy [`.env.example`](../.env.example) to `.env` in the repo root (gitignored) and fill in the provider you use:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `anthropic`, `vertex`, or `openai` |
| `ANTHROPIC_API_KEY` | Direct Anthropic |
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_VERTEX_PROJECT_ID` / `GOOGLE_CLOUD_PROJECT` | Vertex project |
| `CLOUD_ML_REGION` / `GOOGLE_CLOUD_REGION` | Vertex region (often `global`) |
| `LLM_MODEL` | Optional override (blank = Haiku 5.5 / gpt-4.1) |
| `MERCARI_SOURCE` | `auto`, `dpop`, `browser`, or `fixture` |

Vertex uses Application Default Credentials (`gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS`), not `ANTHROPIC_API_KEY`.

## Local run

Streamlit is installed by default with `requirements.txt` / `pip install -e .`:

```bash
streamlit run web/app.py
```

Open http://localhost:8501. Check **Offline fixtures** to search recorded listings without hitting Mercari. Leave it unchecked for live search (home/office IP). Provider and model fields default to the environment; set them only to override.

If the request is too vague, the agent asks one clarifying question. Answer it in the follow-up box and continue. The second pass uses the same clarification prompt as the CLI.

## Docker

The image contains code only. Pass secrets at runtime.

```bash
docker build -t mercari-ai-shopper .
docker run --rm -p 8501:8501 --env-file .env mercari-ai-shopper
```

Or, from the repo root after copying `.env.example` to `.env`:

```bash
docker compose up --build
```

### Vertex inside Docker

Mount ADC into the container. Do not copy a key into the image.

```bash
docker run --rm -p 8501:8501 --env-file .env \
  -e LLM_PROVIDER=vertex \
  -e GOOGLE_APPLICATION_CREDENTIALS=/creds/adc.json \
  -v "$GOOGLE_APPLICATION_CREDENTIALS:/creds/adc.json:ro" \
  mercari-ai-shopper
```

On Windows PowerShell, replace the volume with a path to your ADC JSON, for example:

```powershell
docker run --rm -p 8501:8501 --env-file .env `
  -e LLM_PROVIDER=vertex `
  -e GOOGLE_APPLICATION_CREDENTIALS=/creds/adc.json `
  -v "$env:APPDATA\gcloud\application_default_credentials.json:/creds/adc.json:ro" `
  mercari-ai-shopper
```

## Offline vs live

- **Offline fixtures**: no Mercari network calls. Needs a working LLM key. Use this for demos on cloud IPs and for tests.
- **Live search**: DPoP-signed calls to Mercari from this machine. Works on a typical home or office connection.

**Mercari cloud-IP warning:** a container on AWS, GCP, or GitHub-hosted runners will often get HTTP 403 from Mercari. LLM keys still work there; search may need Offline fixtures / `MERCARI_SOURCE=fixture`.
