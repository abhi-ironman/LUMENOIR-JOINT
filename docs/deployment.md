# LUMENOIR deployment guide

This guide publishes the existing React + FastAPI application without merging
`integration/lumenoir-research` into `main`.

## Hosting layout

- **Frontend:** Vercel, project root `frontend/`, Vite static build.
- **Backend:** Railway, repository root, root `Dockerfile`.
- **Git source for both:** `integration/lumenoir-research`.
- **Model:** published Zero-DCE++ Epoch99 checkpoint downloaded during the
  backend image build and checked against its documented SHA-256.
- **Initial compute:** CPU / float32. The deployment does not request paid GPU
  hardware.

The frontend reads `VITE_API_BASE_URL` at build time. Set it to the public
Railway backend origin (no trailing slash). FastAPI CORS reads
`LUMENOIR_CORS_ORIGINS`, a comma-separated list of exact frontend origins.

## 1. Deploy the backend to Railway

1. Create a Railway project and add a service from GitHub.
2. Select `abhi-ironman/LUMENOIR-JOINT`.
3. Select branch `integration/lumenoir-research`.
4. Set the service root directory to the repository root (`/` or leave it
   unset). Railway must see the root `Dockerfile`, root
   `requirements-inference.txt`, and `backend/`.
5. Deploy using the Dockerfile builder.
6. Set the service healthcheck path to `/api/health` and timeout to
   `300` seconds. The container listens on Railway's injected `PORT`.
7. Generate a public domain for the backend.

The Dockerfile installs `requirements-inference.txt`, downloads the official
Zero-DCE++ Epoch99 checkpoint from the upstream project, verifies its SHA-256,
and enables the model on CPU. The checkpoint is not stored in this repository.
The service uses one Uvicorn worker by default; do not increase workers without
measuring memory use because each worker would load its own model.

Set this Railway variable after the Vercel domain exists:

```text
LUMENOIR_CORS_ORIGINS=https://<your-vercel-domain>
```

Other settings are supplied by the Dockerfile: model ID, checkpoint path,
CPU device, float32 precision, model startup loading and the Phase Engine
disabled by default.

## 2. Deploy the frontend to Vercel

1. Import `abhi-ironman/LUMENOIR-JOINT` as a Vercel project.
2. Set **Root Directory** to `frontend`.
3. Use the Vite framework preset.
4. Build command: `npm run build`.
5. Output directory: `dist`.
6. Connect the Git branch `integration/lumenoir-research`.
7. For a shareable public demo without merging `main`, set the Vercel
   project's Production Branch to `integration/lumenoir-research`. Otherwise,
   the branch deployment remains a Preview deployment.
8. Generate or assign the public Vercel domain.

Set this Vercel environment variable for the environment being published:

```text
VITE_API_BASE_URL=https://<your-railway-domain>
```

Redeploy the frontend after adding or changing this variable; Vite embeds it
at build time.

## 3. Connect the two services

After both public domains exist:

- On Railway, set `LUMENOIR_CORS_ORIGINS` to the exact Vercel origin,
  including `https://` and excluding any path or trailing slash.
- On Vercel, set `VITE_API_BASE_URL` to the Railway origin, including
  `https://` and excluding any path or trailing slash.
- Redeploy both services after updating their variables.

Do not use a wildcard CORS origin for the public deployment.

## 4. Verify the live application

1. Open `https://<your-railway-domain>/api/health`; expect HTTP 200 and
   `status: ok`.
2. Open `https://<your-railway-domain>/api/enhance/status`; verify that the
   Zero-DCE++ model is ready.
3. Open the Vercel domain and upload a small JPG or PNG.
4. Run Classical, Deep Learning and Hybrid modes separately.
5. Confirm the original, intermediate artifacts, final output and download
   are returned correctly.
6. Check the browser network panel for CORS errors and Railway logs for
   inference failures.

## Operational and licensing notes

- This is an academic/demo deployment, not a hardened public production
  service. It has no account system, usage quotas or distributed rate limiter.
  Keep the demo small and monitor usage; disable public access or scale down
  the backend when not presenting.
- The API's configured upload cap is 10 MB. Do not upload private or sensitive
  photographs to a public demo.
- The published Zero-DCE++ checkpoint is documented as CC BY-NC 4.0 in
  `docs/zero_dce.md`. This configuration is for non-commercial academic
  demonstration. Review the model and dataset licences before commercial
  use or redistribution.
- The deployment uses the published pretrained checkpoint, not the
  preliminary LOL-v1 fine-tuned checkpoint. The evaluation report keeps those
  experiments distinct.
- Railway CPU inference may be slower than the local Apple Silicon MPS run.
  Record live response times before advertising performance.
- Vercel and Railway bill separately. Review each provider's current plan and
  usage before enabling paid resources. Do not enable a GPU for this demo
  without explicitly deciding to incur hourly costs.
