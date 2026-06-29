# src/utils/config_loader.py
from collections.abc import Mapping
from pathlib import Path

import yaml
from loguru import logger

# 💎 显式声明：合并后的 config 必须具备哪些字段，写成"路径"的形式
REQUIRED_CONFIG_PATHS = [
    # train
    "train.epochs",
    "train.early_stop.monitor",
    "train.early_stop.mode",
    "train.checkpoint.monitor",
    "train.checkpoint.mode",        # 💎 补：你只查了 monitor，没查 mode，但这俩必须配套（min/max 不一致会让 best model 选反）

    # preprocessing
    "preprocessing.seed",
    "preprocessing.k_fold",
    "preprocessing.batch_size",     # 💎 补：batch_size 没了，DataLoader 可能用一个隐藏默认值跑

    # model —— 第一次那个 bug（model 选错）schema check 完全没覆盖到，必须补
    "model.name",
    "model.backbone",

    # task_type —— 这是你第一次最致命的 bug 本体，必须显式校验存在
    "task_type",

    # dataset_info —— main.py 后面直接用它初始化 DataEDA，缺了会在很晚的阶段才崩
    "dataset_info",
]

KNOWN_TOP_LEVEL_KEYS = {
    "system", "preprocessing", "logging", "train", "evaluation",
    "model", "loss", "task_type", "dataset_name", "dataset_info",
}

def warn_orphan_keys(config: dict) -> None:
    orphans = set(config.keys()) - KNOWN_TOP_LEVEL_KEYS
    if orphans:
        logger.warning(f"⚠️ [Config] 发现未被任何已知字段消费的顶层 key: {orphans}，请检查是否命名拼写错误")


def _get_nested(d: dict, dotted_key: str):
    """按 'a.b.c' 的形式取嵌套字典的值，取不到返回 _MISSING"""
    cur = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def validate_config_schema(config: dict) -> None:
    missing = []
    for path in REQUIRED_CONFIG_PATHS:
        _, found = _get_nested(config, path)
        if not found:
            missing.append(path)
    if missing:
        logger.error(f"❌ [Config Schema] 合并后的配置缺少以下关键字段: {missing}")
        logger.error("   常见原因：basic.yaml 和 model yaml 的顶层 key 命名不一致（如 training_defaults vs train）")
        raise KeyError(f"Config schema validation failed, missing: {missing}")


# Automatically scan config/models/*.yaml
def discover_models(config_dir: Path) -> dict:
    """Automatically discover configuration files in the models directory"""
    models_dir = config_dir / "models"
    if not models_dir.exists():
        return {}

    model_map = {}
    for yaml_file in models_dir.glob("*.yaml"):
        # resnet_iqa.yaml -> resnet_iqa
        model_name = yaml_file.stem
        model_map[model_name] = f"models/{yaml_file.name}"
    return model_map


MODEL_MAP = None  # Lazy loading


def get_model_map(config_dir: Path) -> dict:
    global MODEL_MAP
    if MODEL_MAP is None:
        MODEL_MAP = discover_models(config_dir)
        # Manual mapping as backup
        MODEL_MAP.update({"resnet_iqa": "models/resnet_iqa.yaml", "timeswin_vqa": "models/timeswin_vqa.yaml"})
    return MODEL_MAP


def deep_update(source, overrides):
    """Recursively merge dictionaries, including nested dictionaries."""
    for k, v in overrides.items():
        if isinstance(v, Mapping) and v:
            source[k] = deep_update(source.get(k, {}), v)
        else:
            source[k] = v
    return source


def safe_load_yaml(path: Path, description: str = "configuration file") -> dict:
    """Safely load YAML files, throw an exception on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            if content is None:
                raise ValueError(f"{description} is empty: {path}")
            return content
    except FileNotFoundError as e:
        logger.error(f"❌ {description} does not exist: {path}")
        raise FileNotFoundError(f"{description} does not exist: {path}") from e
    except yaml.YAMLError as e:
        logger.error(f"❌ YAML formatting error: {path}")
        logger.error(f"   Error details: {e}")
        # Help: Check YAML syntax, especially indentation and special characters.
        raise RuntimeError(f"YAML formatting error: {path}") from e
    except Exception as e:
        logger.error(f"❌ Failed to read {description}: {path}, Error: {e}")
        raise


def load_system_config(model_cfg_name: str, dataset_name: str) -> dict:
    config_dir = Path("config")
    config = safe_load_yaml(config_dir / "basic.yaml", "Basic configuration file")

    model_key = str(model_cfg_name).strip().lower()
    model_map = get_model_map(config_dir)

    if model_key not in model_map:
        logger.error(
            f"❌ [Config] 未知的 model_key='{model_key}'，可用值: {list(model_map.keys())}"
        )
        raise ValueError(f"Unknown model_key '{model_key}'. Did you mean one of {list(model_map.keys())}?")


    target_model_file = model_map.get(model_key, "models/resnet_iqa.yaml")
    model_path = config_dir / target_model_file

    if not model_path.exists():
        logger.warning(f"⚠️ Model config [{model_path}] not found. Falling back to resnet_iqa.yaml")
        model_path = config_dir / "models/resnet_iqa.yaml"

    model_config = safe_load_yaml(model_path, f"Model configuration file [{model_key}]")
    config = deep_update(config, model_config)

    warn_orphan_keys(config)   # 💎 补上这一行！merge 完立刻检查孤儿 key

    # 3. Load dataset configuration
    dataset_cfg_path = config_dir / "dataset_config.yaml"
    ds_all = safe_load_yaml(dataset_cfg_path, "Dataset configuration file")

    ds_all_lowered = {k.lower(): v for k, v in ds_all.items()}
    target_ds_key = str(dataset_name).strip().lower()

    if target_ds_key not in ds_all_lowered:
        available = list(ds_all.keys())
        logger.error(f"❌ Dataset '{dataset_name}' not found in dataset_config.yaml")
        logger.info(f"   Available datasets: {available}")
        raise KeyError(f"Dataset settings for '{dataset_name}' missing. Available: {available}")

    dataset_info = ds_all_lowered[target_ds_key]

    # Command-line arguments are prioritized; the name in YAML is not used.
    config["dataset_info"] = dataset_info
    config["dataset_name"] = dataset_name.lower()  # Command line arguments to lowercase

    logger.info(f"⚙️ [Config Engine] Layered configuration successfully built for Model [{model_key}] & Dataset [{config['dataset_name']}]")

    logger.debug(f"[Config] Model configuration: {model_key}, Dataset: {config['dataset_name']}")
    logger.debug(f"[Config] Training configuration: epochs={config.get('train', {}).get('epochs', 'N/A')}")


    validate_config_schema(config)   # 💎 合并完立刻校验，而不是等训练跑了几个小时才发现
    return config
