# src/data/eda/split.py
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_train_val_test(
    df: pd.DataFrame, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1,
    random_state: int = 42,
    score_col: str = "mos"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    划分训练/验证/测试集（分层采样）

    Args:
        df: 包含评分列的 DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_state: 随机种子
        score_col: 评分列名称

    Returns:
        (train_df, val_df, test_df)
    """
    test_ratio = 1.0 - train_ratio - val_ratio

    if test_ratio <= 0:
        raise ValueError(f"无效比例: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    logger.info(f"📊 [Split] 按比例切分: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")

    # 创建分层标签
    stratify_labels = create_stratified_labels(df, score_col=score_col)

    # 第一次划分：分出测试集
    train_val, test = train_test_split(
        df, 
        test_size=test_ratio, 
        random_state=random_state, 
        stratify=stratify_labels
    )

    # 第二次划分：从 train_val 中分出验证集
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_val_stratify = stratify_labels.iloc[train_val.index]
    
    train, val = train_test_split(
        train_val,
        test_size=relative_val_ratio,
        random_state=random_state,
        stratify=train_val_stratify,
    )

    logger.info(f"✅ 划分完成: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    return train, val, test


def create_stratified_labels(df: pd.DataFrame, bins: int = 10, score_col: str = "mos"):
    """
    创建分层标签
    优先使用分位数，失败时回退到均匀分箱
    """
    try:
        return pd.qcut(df[score_col], q=bins, labels=False, duplicates="drop")
    except Exception as e:
        logger.debug(f"分位数切分失败 ({e})，回退到均匀分箱")
        return pd.cut(df[score_col], bins=bins, labels=False)


def check_fold_distribution(
    df: pd.DataFrame, 
    n_splits: int = 5, 
    random_state: int = 42,
    score_col: str = "mos",
    verbose: bool = True
) -> List[Dict]:
    """
    检查K折交叉验证的分布

    Args:
        df: 包含评分列的 DataFrame
        n_splits: K折数量
        random_state: 随机种子
        score_col: 评分列名称
        verbose: 是否输出日志

    Returns:
        每折的统计信息列表
    """
    if df is None or df.empty:
        return []

    stratify_labels = create_stratified_labels(df, score_col=score_col)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_stats = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, stratify_labels)):
        train_mean = df.iloc[train_idx][score_col].mean()
        val_mean = df.iloc[val_idx][score_col].mean()
        train_std = df.iloc[train_idx][score_col].std()
        val_std = df.iloc[val_idx][score_col].std()

        fold_stats.append({
            "fold": fold + 1,
            "train_mean": train_mean,
            "val_mean": val_mean,
            "train_std": train_std,
            "val_std": val_std,
        })

    if verbose:
        logger.info(f"✅ Cross-Validation -> Stratified {n_splits}-Fold distribution checked")
        # 输出统计信息
        for stat in fold_stats:
            logger.debug(
                f"  Fold {stat['fold']}: Train Mean={stat['train_mean']:.3f}±{stat['train_std']:.3f}, "
                f"Val Mean={stat['val_mean']:.3f}±{stat['val_std']:.3f}"
            )

    return fold_stats


def split_from_config(
    df: pd.DataFrame, 
    config: Dict,
    score_col: str = "mos"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    从 YAML 配置读取划分比例
    """
    split_cfg = config.get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.8)
    val_ratio = split_cfg.get("val_ratio", 0.1)
    
    return split_train_val_test(df, train_ratio, val_ratio, score_col=score_col)


def add_split_column(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_state: int = 42,
    score_col: str = "mos"
) -> pd.DataFrame:
    """
    添加 'split' 列到 DataFrame
    
    Returns:
        添加了 'split' 列的 DataFrame (train/val/test)
    """
    train, val, test = split_train_val_test(df, train_ratio, val_ratio, random_state, score_col)
    
    df_copy = df.copy()
    df_copy["split"] = "test"  # 默认都是 test
    df_copy.loc[train.index, "split"] = "train"
    df_copy.loc[val.index, "split"] = "val"
    
    return df_copy