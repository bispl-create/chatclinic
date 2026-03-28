# AYQ Detection Runtime

This plugin now contains a local AYQ runtime under:

- `plugins/detection_ayq/ayq_runtime/mmdet`
- `plugins/detection_ayq/ayq_runtime/text_embeddings/single_class_embeddings_openclip`
- `plugins/detection_ayq/ayq_runtime/open_clip.py` (compatibility shim)

## 1) Export plugin-local runtime env

```bash
cd /path/to/chatclinic
source plugins/detection_ayq/scripts/export_local_runtime_env.sh
```

This sets `AYQ_ROOT` and `AYQ_PYTHON_EXECUTABLE`.

## 2) Start backend in your ChatClinic env

```bash
cd /path/to/chatclinic
uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
```

Keep this shell session (with exported env vars) for backend startup.

## Notes

- The compatibility shim `open_clip.py` is sufficient for this precomputed-embedding inference flow.
- If a future workflow uses on-the-fly text encoding, install `open_clip_torch` in the `ayq` env.
