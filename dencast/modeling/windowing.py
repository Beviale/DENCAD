"""Turning a split directory into windowed feature frames. Activate the Spark session."""

from __future__ import annotations

import os
import sys
from functools import reduce
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import math

from loguru import logger

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    LongType
)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql import Window as SparkWindow



PARTS = ("train", "validation", "test")
INDEX = "datetime"


def session(app: str, workers: int = 4):
    """A local Spark session configured for this environment."""

    # The Python workers are launched by the JVM and have to find the same
    # interpreter this process runs under, or they fail to connect back.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    spark = (SparkSession.builder
             .appName(app)
             .master(f"local[{workers}]")
             .config("spark.driver.host", "127.0.0.1")
             .config("spark.driver.bindAddress", "127.0.0.1")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.python.worker.reuse", "true")
             .config("spark.network.timeout", "600s")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.sql.execution.arrow.maxRecordsPerBatch", "20000")
             # pandas writes the index as TIMESTAMP(NANOS), which Spark 3.5 refuses
             # outright. Read as a bigint of nanoseconds and convert where needed.
             .config("spark.sql.legacy.parquet.nanosAsLong", "true")
             .config("spark.sql.codegen.maxFields", "30")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def discover(split_dir: Path) -> tuple[str, dict[str, Path]]:
    """The dataset name and the three files."""
    found: dict[str, Path] = {}
    names = set()
    for p in PARTS:
        hits = sorted(split_dir.glob(f"*_{p}.parquet"))
        if not hits:
            raise FileNotFoundError(f"no *_{p}.parquet in {split_dir}")
        if len(hits) > 1:
            raise ValueError(f"{len(hits)} candidates for '{p}' in {split_dir}: "
                             f"{', '.join(h.name for h in hits)}")
        found[p] = hits[0]
        names.add(hits[0].name[: -len(f"_{p}.parquet")])
    if len(names) != 1:
        raise ValueError(f"the three files disagree on the dataset name: {names}")
    return names.pop(), found


def feature_columns(columns: Sequence[str]) -> list[str]:
    """Everything that is neither the index nor a label."""
    return [c for c in columns
            if c != INDEX and not c.startswith("is_") and not c.startswith("label_")]


def label_column(columns: Sequence[str]) -> str:
    """The single 'is_*' column to score against."""
    flags = [c for c in columns if c.startswith("is_") or c.startswith("label_")]
    if len(flags) != 1:
        raise ValueError(f"expected exactly one 'is_*' column, found {flags}")
    return flags[0]


def _to_seconds(df, column: str = INDEX):
    """The index as seconds, whichever of the two forms Spark read it in."""

    if dict(df.dtypes)[column] == "bigint":
        return F.col(column) / F.lit(1_000_000_000)
    return F.unix_timestamp(F.col(column))


def window_pandas(df, columns: list[str], seconds: int, label: str,
                  categorical: list[str]):
    numeric = [c for c in columns if c not in set(categorical)]
    
    base = df.withColumn("_w", F.floor(_to_seconds(df) / seconds) * seconds)

    schema_fields = [
        StructField("_w", LongType(), True),
        StructField(label, DoubleType(), True),
        StructField("_rows", LongType(), True)
    ]
    
    names = []
    for c in numeric:
        mean_col = f"{c}_mean"
        std_col = f"{c}_std"
        names.extend([mean_col, std_col])
        schema_fields.append(StructField(mean_col, DoubleType(), True))
        schema_fields.append(StructField(std_col, DoubleType(), True))
        
    if categorical:
        filled = [f"{c}_{k}" for c in categorical for k in ("mode", "entropy", "nunique", "maxprop")]
        names.extend(filled)
        for col_name in filled:
            schema_fields.append(StructField(col_name, DoubleType(), True))
            
    out_schema = StructType(schema_fields)

    def process_window(key: tuple, pdf: pd.DataFrame) -> pd.DataFrame:
        w_val = key[0]
        row_count = len(pdf)
        
        row_dict = {
            "_w": w_val,
            label: float(pdf[label].max()) if label in pdf else 0.0,
            "_rows": row_count
        }
        
        for c in numeric:
            vals = pdf[c].dropna()
            if len(vals) > 0:
                row_dict[f"{c}_mean"] = float(vals.mean())
                std_val = float(vals.std(ddof=0))
                row_dict[f"{c}_std"] = 0.0 if np.isnan(std_val) else std_val
            else:
                row_dict[f"{c}_mean"] = 0.0
                row_dict[f"{c}_std"] = 0.0
                logger.warning(f"The numeric column {c} does not contain any value!")

        if categorical:
            for c in categorical:
                s = pdf[c].dropna()
                if len(s) == 0:
                    logger.warning(f"The categorical column {c} does not contain any value!")
                    row_dict[f"{c}_mode"] = 0.0
                    row_dict[f"{c}_entropy"] = 0.0
                    row_dict[f"{c}_nunique"] = 0.0
                    row_dict[f"{c}_maxprop"] = 0.0
                else:
                    counts = s.value_counts()
                    n_total = counts.sum()
                    
                    row_dict[f"{c}_mode"] = float(counts.idxmax())
                    row_dict[f"{c}_nunique"] = float(len(counts))
                    row_dict[f"{c}_maxprop"] = float(counts.max() / n_total)
                    
                    probs = counts / n_total
                    entropy_val = -float((probs * np.log(probs)).sum())
                    if np.isnan(entropy_val):                     
                        logger.warning(f"The calculated entropy is not valid!")
                    row_dict[f"{c}_entropy"] = 0.0 if np.isnan(entropy_val) else entropy_val

        return pd.DataFrame([row_dict])

    out = base.groupBy("_w").applyInPandas(process_window, schema=out_schema)

    out = out.withColumn(INDEX, F.timestamp_seconds("_w")).drop("_w")

    return out, names


def sampling_period(df, column: str = INDEX) -> int:
    """The seconds between consecutive readings, taken as the commonest gap."""
    from pyspark.sql import Window as SparkWindow

    secs = _to_seconds(df, column)
    gaps = (df.select(secs.alias("_t"))
            .withColumn("_prev", F.lag("_t").over(SparkWindow.orderBy("_t")))
            .filter(F.col("_prev").isNotNull())
            .withColumn("_gap", (F.col("_t") - F.col("_prev")).cast("long"))
            .groupBy("_gap").agg(F.count(F.lit(1)).alias("_n"))
            .orderBy(F.col("_n").desc()).limit(8).collect())
    if not gaps:
        raise ValueError("a single reading cannot say how often the stream samples")
    total = sum(int(r["_n"]) for r in gaps)
    best = max(gaps, key=lambda r: int(r["_n"]))
    step, share = int(best["_gap"]), int(best["_n"]) / max(total, 1)
    if step <= 0:
        raise ValueError("the commonest gap between readings is zero: duplicate "
                         "timestamps reached the modelling stage")
    if share < 0.9:
        logger.warning("  the commonest gap of {} s covers only {:.1%} of consecutive "
                       "readings: this stream is not on a regular grid and a window "
                       "in steps will not be a window in time", step, share)
    else:
        logger.info("  sampling period {} s ({:.2%} of consecutive readings)",
                    step, share)
    return step


def window(df, columns: list[str], steps: int, label: str,
           categorical: list[str], min_coverage: float = 0.0,
           period: Optional[int] = None):
    """Aggregate the rows into windows of 'steps' readings.
    """
    numeric = [c for c in columns if c not in set(categorical)]
    period = int(period or sampling_period(df))
    seconds = int(steps) * period
    base = df.withColumn("_w", F.floor(_to_seconds(df) / seconds) * seconds)

    names: list[str] = []
    aggs = [F.max(F.col(label)).cast("double").alias(label),
            F.count(F.lit(1)).alias("_rows")]
    for c in numeric:
        names += [f"{c}_mean", f"{c}_std"]
        aggs.append(F.coalesce(F.mean(F.col(c)), F.lit(0.0)).alias(f"{c}_mean"))
        aggs.append(F.coalesce(F.stddev_pop(F.col(c)), F.lit(0.0)).alias(f"{c}_std"))  
    out = base.groupBy("_w").agg(*aggs)

    if categorical:
        stacked = base.select(
            "_w",
            F.explode(F.array(*[
                F.struct(F.lit(c).alias("_col"), F.col(c).cast("double").alias("_v"))
                for c in categorical])).alias("_s")
        ).select("_w", F.col("_s._col").alias("_col"), F.col("_s._v").alias("_v"))

        counts = (stacked.filter(F.col("_v").isNotNull())
                  .groupBy("_w", "_col", "_v").agg(F.count(F.lit(1)).alias("_n")))
        rolled = counts.groupBy("_w", "_col").agg(
            F.sum("_n").alias("_total"),
            F.max("_n").alias("_top"),
            F.count(F.lit(1)).alias("_nunique"),
            F.sum(F.col("_n") * F.log(F.col("_n").cast("double"))).alias("_nlogn"),
            F.max(F.struct(F.col("_n").alias("n"),
                           F.negate(F.col("_v")).alias("negv"))).alias("_best"))
        summarised = rolled.select(
            "_w", "_col",
            F.negate(F.col("_best.negv")).alias("mode"),
            (F.log(F.col("_total").cast("double"))
             - F.col("_nlogn") / F.col("_total")).alias("entropy"),
            F.col("_nunique").cast("double").alias("nunique"),
            (F.col("_top") / F.col("_total")).alias("maxprop"))

        wide = summarised.groupBy("_w").pivot("_col", categorical).agg(
            F.first("mode").alias("mode"), F.first("entropy").alias("entropy"),
            F.first("nunique").alias("nunique"), F.first("maxprop").alias("maxprop"))
        for c in categorical:
            names += [f"{c}_mode", f"{c}_entropy", f"{c}_nunique", f"{c}_maxprop"]
        out = out.join(wide, "_w", "left")
        for c in categorical:
            for k in ("mode", "entropy", "nunique", "maxprop"):
                out = out.withColumn(f"{c}_{k}",
                                     F.coalesce(F.col(f"`{c}_{k}`"), F.lit(0.0)))

    if min_coverage > 0:
        out = _drop_thin_windows(out, min_coverage, int(steps))

    out = out.withColumn(INDEX, F.timestamp_seconds("_w")).drop("_w")
    return out.select(INDEX, label, "_rows", *names), names


def _drop_thin_windows(out, min_coverage: float, expected: int):
    """Drop windows built from too few of the readings they should have held."""

    dist = out.groupBy("_rows").count().collect()
    if not dist:
        return out
    floor_rows = max(1, math.ceil(min_coverage * expected))
    total = sum(r["count"] for r in dist)
    thin = sum(r["count"] for r in dist if r["_rows"] < floor_rows)
    if not thin:
        logger.info("  every window holds at least {} of the {} readings a complete "
                    "one has ({:.0%} coverage asked)", floor_rows, expected,
                    min_coverage)
        return out
    worst = min(r["_rows"] for r in dist)
    logger.warning("  dropping {:,} of {:,} windows ({:.2%}) holding fewer than {} of "
                   "the {} readings a complete one has; the thinnest held {}",
                   thin, total, thin / total, floor_rows, expected, worst)
    return out.filter(F.col("_rows") >= F.lit(floor_rows))

def number_rows(df, order: str = INDEX, name: str = "_id"):
    ordered = (df.orderBy(order)
                 .withColumn("_part", F.spark_partition_id())
                 .persist())
    within = (SparkWindow.partitionBy("_part").orderBy(order)
              .rowsBetween(SparkWindow.unboundedPreceding, 0))
    ranked = ordered.withColumn("_local", F.count(F.lit(1)).over(within))
    totals = (ordered.groupBy("_part").agg(F.count(F.lit(1)).alias("_total"))
              .orderBy("_part").collect())
    offsets, running = {}, 0
    for row in totals:
        offsets[int(row["_part"])] = running
        running += int(row["_total"] or 0)
    shift = F.create_map(*[lit for part, off in offsets.items()
                           for lit in (F.lit(part), F.lit(off))])
    out = (ranked.withColumn(name, (F.col("_local") - 1 + shift[F.col("_part")])
                             .cast("long"))
           .drop("_part", "_local")
           .localCheckpoint(eager=True))
    ordered.unpersist()
    return out


SCALE_SKIP = ("_mode",)


def scalable(names: Sequence[str], skip: Sequence[str] = SCALE_SKIP) -> list[str]:
    """The window features a z-score applies to."""
    return [c for c in names if not c.endswith(tuple(skip))]


def window_statistics(train, names: Sequence[str],
                      skip: Sequence[str] = SCALE_SKIP) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of each scalable window feature, on train alone."""
    cols = scalable(names, skip)
    if not cols:
        return {}
    aggs = []
    for c in cols:
        aggs += [F.mean(F.col(c)).alias(f"m_{c}"), F.stddev(F.col(c)).alias(f"s_{c}")]
    row = train.agg(*aggs).first()

    stats, flat = {}, []
    for c in cols:
        s = row[f"s_{c}"]
        s = 0.0 if s is None else float(s)
        if s == 0.0:
            flat.append(c)
            s = 1.0
        stats[c] = (float(row[f"m_{c}"] or 0.0), s)
    if flat:
        logger.warning("  {} window features have zero spread in training; centred but "
                       "not scaled: {}", len(flat), ", ".join(flat[:8]))
    skipped = len(names) - len(cols)
    logger.info("  window scaling fitted on train: {} features z-scored, {} left as "
                "given ({})", len(cols), skipped, ", ".join(skip))
    return stats


def apply_scaling(df, stats: dict[str, tuple[float, float]]):
    if not stats:
        return df
    out = []
    for c in df.columns:
        if c in stats:
            mean, std = stats[c]
            out.append(((F.col(c) - F.lit(mean)) / F.lit(std)).alias(c))
        else:
            out.append(F.col(c))
    return df.select(*out)


def prepare(spark, files: dict[str, Path], categorical: Sequence[str],
            window_steps: int, parts: Sequence[str] = PARTS,
            features: Optional[Sequence[str]] = None, scale: bool = True,
            scaling: Optional[dict[str, tuple[float, float]]] = None,
            min_coverage: float = 0.0) -> dict:
    """Read the parts, window them, scale them, assemble the feature vector."""

    raw = {p: spark.read.parquet(str(files[p].resolve())) for p in parts}
    first = raw[parts[0]]
    columns = feature_columns(first.columns)
    label = label_column(first.columns)
    rows = {p: raw[p].count() for p in parts}

    cat = [c for c in columns if c in set(categorical)]
    logger.info("label column '{}' | {} numeric, {} categorical: {}",
                label, len(columns) - len(cat), len(cat), ", ".join(cat) or "none")

    if not window_steps or window_steps <= 1:
        raise ValueError(
            f"window_steps must be greater than 1, got {window_steps!r}: the "
            "features of this project are window summaries, and one row per instant "
            "is not a space any of the models is defined in")
  
    period = sampling_period(raw[parts[0]])
    frames, built = {}, None
    for p in parts:
        logger.info(f"Creating the windows for the {p} split...")
        frames[p], built = window(raw[p], columns, window_steps, label, cat,
                                  min_coverage=min_coverage, period=period)
    logger.info("windowed at {} steps of {} s: {} columns -> {} features (mean and "
                "std of each numeric, mode/entropy/nunique/maxprop of each "
                "categorical)", window_steps, period, len(columns), len(built))

    if scaling is None and scale:
        scaling = window_statistics(frames["train"], built)
    if scaling:
        frames = {p: apply_scaling(frames[p], scaling) for p in parts}

    if features is not None:
        if list(features) != list(built):
            missing = sorted(set(features) - set(built))
            extra = sorted(set(built) - set(features))
            raise ValueError(
                "the windowing does not reproduce the feature space the model was "
                f"fitted in: {len(missing)} expected columns are absent "
                f"({', '.join(missing[:5]) or 'none'}) and {len(extra)} are new "
                f"({', '.join(extra[:5]) or 'none'}); same names in a different order "
                "counts too, because a centroid's coordinates are positional")
        built = list(features)

    assembler = VectorAssembler(inputCols=list(built), outputCol="features")
    vec = {p: assembler.transform(frames[p])
                       .select(INDEX, "features", label).cache()
           for p in parts}
    return {"vec": vec, "features": list(built), "label": label,
            "columns": columns, "rows": rows, "scaling": scaling}



__all__ = ["PARTS", "INDEX", "SCALE_SKIP", "session", "discover", "feature_columns",
           "label_column", "window", "window_pandas", "scalable", "window_statistics",
           "apply_scaling", "prepare"]
