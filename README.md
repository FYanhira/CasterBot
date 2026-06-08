# CASTERBOT

Pipeline [SAM 3](https://github.com/facebookresearch/sam3) para la Copa FutBotMX (Visión por Computadora, categoría Profesional): segmentación, tracking, eventos tácticos y transmisión automática con overlays.

## Estado (Fase 1)

Pipeline implementado: segmentación SAM 3 → `tracks.json` → `events.json` → `script.json` → video `original | overlay`.

Documentación de diseño (repo `futbot`): [`../docs/tactibot/`](../docs/tactibot/)

## Videos de trabajo

**No** leas carpetas externas (`17Abril`, `18abril`) desde el código.

1. Copia **uno o pocos** `.mp4` a [`data/videos/`](data/videos/).
2. Ajusta `video_path` en `configs/default.yaml`.

Ver [`data/videos/README.md`](data/videos/README.md).

## Token de Hugging Face (cuándo y dónde)

Necesitas acceso a [facebook/sam3](https://huggingface.co/facebook/sam3). El modelo descarga **`config.json`** + **`sam3.pt`** (~3.5 GB) la **primera vez** que se construye el predictor.

### Opción A — Recomendada (una vez por máquina)

```bash
pip install huggingface_hub
huggingface-cli login
# Pega tu token cuando lo pida (Settings → Access Tokens en huggingface.co)
```

No hace falta poner el token en el código. Se guarda en `~/.cache/huggingface/`.

### Opción B — Variable de entorno (CI o sesión actual)

```bash
cp .env.example .env
# Edita .env y pon HF_TOKEN=hf_...
export HF_TOKEN="hf_..."
```

`huggingface_hub` lee **`HF_TOKEN`** (o `HUGGING_FACE_HUB_TOKEN`) automáticamente al descargar.

### Momento exacto de uso

| Paso | ¿Usa el token? |
|------|----------------|
| `pip install`, `git clone` | No |
| Copiar video a `data/videos/` | No |
| **`tactibot run`** → primera llamada a `build_sam3_predictor()` | **Sí** — descarga checkpoint si no está en caché |
| Siguientes ejecuciones | No (usa caché local salvo que borres `~/.cache/huggingface`) |

**Nunca** commitees el token. `.env` está en `.gitignore`.

## Instalación

Requisitos: Python 3.12+, NVIDIA GPU, CUDA 12.6+, PyTorch 2.7+ (ver [sam3 README](../sam3/README.md)).

```bash
git clone https://github.com/FYanhira/CASTERBOT.git
cd CASTERBOT

# Entorno
conda create -n casterbot python=3.12 -y
conda activate casterbot

# PyTorch CUDA (ajusta según tu driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# SAM 3 (dependencia Meta; no hace falta clonar el repo)
pip install git+https://github.com/facebookresearch/sam3.git

# CASTERBOT
pip install -e .

# Autenticación HF (elige A o B arriba)
huggingface-cli login
```

## Uso

```bash
# 1. Coloca tu video
cp /ruta/a/tu/partido.mp4 data/videos/partido_mvp.mp4

# 2. Prueba corta (90 frames) — edita video_path en configs/example_short.yaml
tactibot run --config configs/example_short.yaml

# 3. Partido completo
tactibot run --config configs/default.yaml
```

Salidas en `outputs/<run_name>/`:

- `tracks.json` — detecciones por frame  
- `events.json` — eventos heurísticos  
- `script.json` — narración template  
- `renders/demo_side_by_side.mp4` — original | overlay  

## Estructura

```text
CASTERBOT/
├── tactibot/          # Paquete Python del proyecto
├── configs/           # YAML
├── data/videos/       # Tus videos (mp4 ignorados por git)
└── outputs/           # Artefactos (ignorado por git)
```

## Licencia

MIT — ver [LICENSE](LICENSE). Uso de SAM 3 sujeto a [SAM License](https://github.com/facebookresearch/sam3).

## Contacto reto

`futbotmx@secihti.mx` · [secihti.mx/futbotmx](https://secihti.mx/futbotmx/)
