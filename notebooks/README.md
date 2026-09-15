# notebooks/

Exploration only. Import the installed package (`retention_radar`); **do not** put production train/serve logic here.

```python
# After: pip install -e .   (or PYTHONPATH=src)
from retention_radar import config
from retention_radar.data.generate import generate_users
```

Promoted code belongs under `src/retention_radar/`. Charts for the published narrative live in `results/plots/`, not notebook outputs.
