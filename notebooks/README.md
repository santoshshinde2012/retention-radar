# notebooks/

Exploration only. Import the installed package (`retention_radar`); **do not** put production train/serve logic here.

```python
# After: pip install -e .   (or PYTHONPATH=src)
from retention_radar import config
from retention_radar.data.generate import generate_users
```

| Notebook | Purpose |
|----------|---------|
| [`01_explore_santosh.ipynb`](01_explore_santosh.ipynb) | Load Santosh JSON + score with committed seed-42 models |

Promoted code belongs under `src/retention_radar/`. Charts for the published narrative live in `results/plots/`, not notebook outputs.
