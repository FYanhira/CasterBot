# Videos de trabajo (CASTERBOT)

Coloca aquí **solo los videos que usarás** en el pipeline (copiados desde tus carpetas locales `17Abril`, `18abril`, etc.).

- El código **solo** lee desde `data/videos/` (o la ruta que indiques en `configs/*.yaml`).
- **No** commitees videos grandes: están ignorados por `.gitignore` (`*.mp4`, etc.).
- Para el MVP basta **un partido** (un archivo `.mp4`).

## Ejemplo

```text
data/videos/
  README.md          ← este archivo (sí en git)
  .gitkeep
  partido_mvp.mp4    ← copia manual; no subir a GitHub si es pesado
```

En `configs/default.yaml` ajusta:

```yaml
video_path: data/videos/partido_mvp.mp4
```
