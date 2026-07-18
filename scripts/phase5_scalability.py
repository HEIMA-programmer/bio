"""阶段五 · 可扩展性曲线：训练时间 / 进程内存 / CUDA 内存随细胞数增长。

动机
    论文 Ext. Data Fig. 4e,f 报告时间与内存随细胞数的扩展趋势，但其进程/设备内存口径
    不能从图中完全还原。这里在本机 4060 上对递增子集训练固定 epoch，分别测墙钟时间、
    fresh-worker 进程 RSS/private 与 PyTorch CUDA allocated/reserved，避免把固定 minibatch 的
    平坦 CUDA allocated 曲线冒充论文的进程总内存曲线。

设计
    - 从 tcell_processed.h5ad 里按 patient 分层子采样出 n ∈ {10k, 30k, 60k, 100k} 细胞。
    - 每个规模由一个全新的 Python worker 进程独立完成；主进程不加载 h5ad，也不复用
      前一规模的 Python/PyTorch allocator，避免固定的全量 AnnData 基线污染内存曲线。
    - worker 先启动进程内存采样，再以 backed='r' 打开 h5ad，只把所选行的 X、
      patient/cell_type 和 var materialize 成最小 AnnData。
    - 每个规模都用**相同的固定 epoch 数**（默认 20）训练官方 scAtlasVAE（监督），
      这样"每细胞成本"才可比（否则 fit() 会按规模自动改 epoch 数、混淆变量）。
    - 记录：fit() 墙钟秒数、进程 RSS/working set 与 private memory 峰值，
      以及 CUDA max_memory_allocated / max_memory_reserved（均为 MiB）。
      进程内存是 Python 进程的总占用，包含 AnnData/CPU 侧对象；CUDA 两列只表示
      PyTorch allocator 的 allocated/reserved，并不等同于进程总显存。
      fit_seconds 只计 model.fit()；setup_and_fit_seconds 计模型初始化+fit；
      load_setup_fit_seconds 从 backed 读取开始计到训练同步结束，是端到端 worker 核心时间。
    - 只读主 h5ad、只在子集上临时训练，**不写回任何 obsm**，与主流水线无冲突。

用法（环境 A `scatlasvae`）
    python phase5_scalability.py                       # 默认 10k/30k/60k/100k，各 20 epoch
    python phase5_scalability.py --sizes 10000 50000   # 自定义规模
    python phase5_scalability.py --epochs 15

产出
    phase5_scalability.csv：保留原有五列，并追加进程内存、CUDA allocated/reserved 等列。

对应报告
    reports/phase5_deeper_validation.md（可扩展性一节，对标 Ext. Data Fig. 4e,f）。
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

PROC_PATH = "tcell_processed.h5ad"
BATCH_KEY = "patient"
LABEL_KEY = "cell_type"
OUT = "phase5_scalability.csv"
MIB = 1024 ** 2
DEFAULT_MEMORY_SAMPLE_INTERVAL_MS = 50.0
RESULT_PREFIX = "SCALABILITY_RESULT_JSON="

# 前五列是旧版文件的完整 schema。保留列名和含义，避免既有作图与报告读取失效；
# peak_gpu_mb 是 peak_cuda_allocated_mb 的兼容别名。
CSV_COLUMNS = [
    "n_cells",
    "fit_seconds",
    "peak_gpu_mb",
    "sec_per_epoch",
    "sec_per_10k_cells",
    "setup_and_fit_seconds",
    "data_load_seconds",
    "runtime_import_seconds",
    "model_setup_seconds",
    "load_setup_fit_seconds",
    "start_process_rss_mb",
    "peak_process_rss_mb",
    "peak_process_rss_delta_mb",
    "start_process_working_set_mb",
    "peak_process_working_set_mb",
    "peak_process_working_set_delta_mb",
    "start_process_private_mb",
    "peak_process_private_mb",
    "peak_process_private_delta_mb",
    "peak_cuda_allocated_mb",
    "peak_cuda_reserved_mb",
    "process_memory_backend",
    "process_memory_samples",
    "process_memory_sample_interval_ms",
    "process_memory_scope",
    "worker_pid",
]


def _make_process_memory_reader():
    """返回 ``(reader, backend)``；reader 给出 RSS、working set、private bytes。

    优先用 psutil。项目环境没有 psutil 时，Windows 上直接调用
    GetProcessMemoryInfo，因此该评测不需要为了三个计数器额外安装依赖。
    非 Windows 且无 psutil 时返回不可用，由 CSV 中的 NaN 明确表示。
    """
    try:
        import psutil

        process = psutil.Process(os.getpid())
        process.memory_info()  # 在启动采样线程前验证句柄可读。

        def read_psutil():
            info = process.memory_info()
            rss = int(info.rss)
            # Windows 的 RSS 就是 working set；部分 psutil 版本另提供 wset。
            working_set = int(getattr(info, "wset", rss))
            private = getattr(info, "private", None)
            if private is None and os.name == "nt":
                private = getattr(process.memory_full_info(), "private", None)
            return rss, working_set, None if private is None else int(private)

        return read_psutil, "psutil"
    except (ImportError, OSError):
        pass

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process_handle = kernel32.GetCurrentProcess()

            def read_windows_ctypes():
                counters = PROCESS_MEMORY_COUNTERS_EX()
                counters.cb = ctypes.sizeof(counters)
                ok = psapi.GetProcessMemoryInfo(
                    process_handle, ctypes.byref(counters), counters.cb
                )
                if not ok:
                    error_code = ctypes.get_last_error()
                    raise OSError(error_code, "GetProcessMemoryInfo failed")
                working_set = int(counters.WorkingSetSize)
                # Windows 中 RSS 与当前 working set 是同一口径。
                return working_set, working_set, int(counters.PrivateUsage)

            read_windows_ctypes()  # 立即验证结构体/API 定义。
            return read_windows_ctypes, "windows-ctypes"
        except (ImportError, OSError, AttributeError):
            pass

    return None, "unavailable"


class ProcessPeakMemoryMonitor:
    """以固定间隔采样当前 Python 进程的 CPU 内存峰值。"""

    def __init__(self, interval_ms=DEFAULT_MEMORY_SAMPLE_INTERVAL_MS):
        if interval_ms <= 0:
            raise ValueError("memory sampling interval must be > 0 ms")
        self.interval_seconds = interval_ms / 1000.0
        self.reader, self.backend = _make_process_memory_reader()
        self.samples = 0
        self.start = [None, None, None]
        self.peak = [None, None, None]
        self._stop_event = threading.Event()
        self._thread = None

    def _sample_once(self):
        if self.reader is None:
            return
        try:
            values = self.reader()
        except OSError:
            # 一次瞬时读取失败不应中断数小时的训练；后续采样仍会重试。
            return
        if self.samples == 0:
            self.start = list(values)
        for i, value in enumerate(values):
            if value is not None and (self.peak[i] is None or value > self.peak[i]):
                self.peak[i] = value
        self.samples += 1

    def _run(self):
        while not self._stop_event.wait(self.interval_seconds):
            self._sample_once()

    def start_sampling(self):
        self._sample_once()
        if self.reader is not None:
            self._thread = threading.Thread(
                target=self._run, name="process-memory-monitor", daemon=True
            )
            self._thread.start()

    def stop_sampling(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self._sample_once()

    @staticmethod
    def _mb(value):
        return float("nan") if value is None else value / MIB

    def summary_mb(self):
        start_rss, start_working_set, start_private = map(self._mb, self.start)
        peak_rss, peak_working_set, peak_private = map(self._mb, self.peak)

        def delta(peak, start):
            if np.isnan(peak) or np.isnan(start):
                return float("nan")
            return max(0.0, peak - start)

        return {
            "start_process_rss_mb": start_rss,
            "peak_process_rss_mb": peak_rss,
            "peak_process_rss_delta_mb": delta(peak_rss, start_rss),
            "start_process_working_set_mb": start_working_set,
            "peak_process_working_set_mb": peak_working_set,
            "peak_process_working_set_delta_mb": delta(peak_working_set, start_working_set),
            "start_process_private_mb": start_private,
            "peak_process_private_mb": peak_private,
            "peak_process_private_delta_mb": delta(peak_private, start_private),
            "process_memory_backend": self.backend,
            "process_memory_samples": self.samples,
            "process_memory_sample_interval_ms": self.interval_seconds * 1000.0,
        }


def stratified_positions(obs, n_target, seed=0):
    """按 patient 分层返回有序行号，目标约为 n_target。"""
    if len(obs) <= n_target:
        return np.arange(len(obs), dtype=np.int64)
    rng = np.random.default_rng(seed)
    fraction = n_target / len(obs)
    selected = []
    groups = obs.groupby(BATCH_KEY, observed=True, sort=False).indices
    for positions in groups.values():
        positions = np.asarray(positions, dtype=np.int64)
        k = max(1, int(round(len(positions) * fraction)))
        selected.extend(
            rng.choice(positions, size=min(k, len(positions)), replace=False).tolist()
        )
    # h5py/backed sparse fancy indexing要求行号递增。
    return np.sort(np.asarray(selected, dtype=np.int64))


def load_minimal_backed_adata(input_path, n_target, seed=0):
    """从 backed h5ad 只 materialize 训练所需字段。"""
    import anndata as ad

    backed = ad.read_h5ad(input_path, backed="r")
    try:
        required = [BATCH_KEY, LABEL_KEY]
        missing = [key for key in required if key not in backed.obs]
        if missing:
            raise KeyError(f"h5ad 缺少必须的 obs 列: {missing}")
        obs_for_sampling = backed.obs.loc[:, required]
        positions = stratified_positions(obs_for_sampling, n_target, seed)
        # positions 已排序，兼容 backed CSRDataset 与 dense h5py Dataset。
        x_selected = backed.X[positions, :]
        if hasattr(x_selected, "copy"):
            x_selected = x_selected.copy()
        obs_selected = obs_for_sampling.iloc[positions].copy()
        var_selected = backed.var.copy()
    finally:
        backed.file.close()

    # anndata 0.8 默认会把整数 counts 再复制成 float32；显式保留源 dtype，既与原 h5ad
    # 一致，也避免一次会随 n 增长的无意 CPU 内存复制。
    return ad.AnnData(
        X=x_selected,
        obs=obs_selected,
        var=var_selected,
        dtype=x_selected.dtype,
    )


def measure_in_fresh_worker(
    input_path,
    n_target,
    epochs,
    memory_sample_interval_ms=DEFAULT_MEMORY_SAMPLE_INTERVAL_MS,
    seed=0,
    load_only=False,
):
    """worker 内执行 backed 读取、模型构造和训练；返回一行 CSV 指标。"""
    monitor = ProcessPeakMemoryMonitor(memory_sample_interval_ms)
    monitor.start_sampling()
    load_setup_fit_t0 = time.perf_counter()
    model = None
    adata_sub = None
    torch_module = None
    peak_cuda_allocated_mb = float("nan")
    peak_cuda_reserved_mb = float("nan")
    runtime_import_seconds = 0.0
    model_setup_seconds = 0.0
    fit_seconds = 0.0
    setup_and_fit_seconds = 0.0
    try:
        data_load_t0 = time.perf_counter()
        adata_sub = load_minimal_backed_adata(input_path, n_target, seed)
        data_load_seconds = time.perf_counter() - data_load_t0
        n_real = int(adata_sub.n_obs)

        if not load_only:
            runtime_import_t0 = time.perf_counter()
            import scatlasvae
            import torch

            torch_module = torch
            runtime_import_seconds = time.perf_counter() - runtime_import_t0
            if not torch.cuda.is_available():
                raise RuntimeError("phase5_scalability.py 需要可用的 CUDA GPU")

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            setup_and_fit_t0 = time.perf_counter()
            model_setup_t0 = time.perf_counter()
            model = scatlasvae.model.scAtlasVAE(
                adata=adata_sub,
                batch_key=BATCH_KEY,
                label_key=LABEL_KEY,
                device="cuda:0",
            )
            torch.cuda.synchronize()
            model_setup_seconds = time.perf_counter() - model_setup_t0
            fit_t0 = time.perf_counter()
            model.fit(max_epoch=epochs)
            # CUDA kernel launch 是异步的；同步后再停表，确保 wall time 覆盖实际训练。
            torch.cuda.synchronize()
            fit_seconds = time.perf_counter() - fit_t0
            setup_and_fit_seconds = time.perf_counter() - setup_and_fit_t0
            peak_cuda_allocated_mb = torch.cuda.max_memory_allocated() / MIB
            peak_cuda_reserved_mb = torch.cuda.max_memory_reserved() / MIB

        load_setup_fit_seconds = time.perf_counter() - load_setup_fit_t0
    finally:
        # 在释放最小 AnnData/模型前停止采样，峰值口径覆盖 load+setup+fit 全阶段。
        monitor.stop_sampling()
        if model is not None:
            del model
        if adata_sub is not None:
            del adata_sub
        if torch_module is not None:
            torch_module.cuda.empty_cache()

    return {
        "n_cells": n_real,
        "fit_seconds": fit_seconds,
        # 旧版列名兼容：此前 peak_gpu_mb 就是 max_memory_allocated。
        "peak_gpu_mb": peak_cuda_allocated_mb,
        "sec_per_epoch": fit_seconds / epochs,
        "sec_per_10k_cells": fit_seconds / (n_real / 10000),
        "setup_and_fit_seconds": setup_and_fit_seconds,
        "data_load_seconds": data_load_seconds,
        "runtime_import_seconds": runtime_import_seconds,
        "model_setup_seconds": model_setup_seconds,
        "load_setup_fit_seconds": load_setup_fit_seconds,
        "peak_cuda_allocated_mb": peak_cuda_allocated_mb,
        "peak_cuda_reserved_mb": peak_cuda_reserved_mb,
        "process_memory_scope": "fresh_worker_load_setup_fit",
        "worker_pid": os.getpid(),
        **monitor.summary_mb(),
    }


def run_worker_subprocess(args, n_target):
    """启动单规模 fresh worker，转发普通输出并解析其 JSON 结果行。"""
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--_worker-size",
        str(n_target),
        "--input",
        os.path.abspath(args.input),
        "--epochs",
        str(args.epochs),
        "--memory-sample-interval-ms",
        str(args.memory_sample_interval_ms),
        "--seed",
        str(args.seed),
    ]
    if args._worker_load_only:
        command.append("--_worker-load-only")
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    result = None
    try:
        for line in process.stdout:
            if line.startswith(RESULT_PREFIX):
                result = json.loads(line[len(RESULT_PREFIX):])
            else:
                print(line, end="", flush=True)
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise

    if return_code != 0:
        raise RuntimeError(f"scalability worker n={n_target} 失败，退出码 {return_code}")
    if result is None:
        raise RuntimeError(f"scalability worker n={n_target} 未返回 JSON 结果")
    missing = [column for column in CSV_COLUMNS if column not in result]
    if missing:
        raise RuntimeError(f"scalability worker n={n_target} 缺少结果列: {missing}")
    return result


def write_results(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_result(row):
    print(
        f"n={row['n_cells']:>6}  fit={row['fit_seconds']:7.1f}s  "
        f"load/setup/fit={row['data_load_seconds']:.1f}/"
        f"{row['model_setup_seconds']:.1f}/{row['fit_seconds']:.1f}s  "
        f"process RSS/private={row['peak_process_rss_mb']:7.0f}/"
        f"{row['peak_process_private_mb']:7.0f} MiB  "
        f"CUDA allocated/reserved={row['peak_cuda_allocated_mb']:7.0f}/"
        f"{row['peak_cuda_reserved_mb']:7.0f} MiB",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10000, 30000, 60000, 100000])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--input", default=PROC_PATH, help="输入 h5ad（默认当前目录 tcell_processed.h5ad）")
    ap.add_argument("--output", default=OUT, help="汇总 CSV 输出路径")
    ap.add_argument("--seed", type=int, default=0, help="patient 分层子采样随机种子")
    ap.add_argument(
        "--memory-sample-interval-ms",
        type=float,
        default=DEFAULT_MEMORY_SAMPLE_INTERVAL_MS,
        help="进程 RSS/working set/private memory 采样间隔（默认 50 ms）",
    )
    ap.add_argument("--_worker-size", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--_worker-load-only", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.epochs <= 0:
        ap.error("--epochs 必须大于 0")
    if any(n <= 0 for n in args.sizes):
        ap.error("--sizes 中的值必须大于 0")
    if args.memory_sample_interval_ms <= 0:
        ap.error("--memory-sample-interval-ms 必须大于 0")
    if args._worker_size is not None and args._worker_size <= 0:
        ap.error("--_worker-size 必须大于 0")
    if not os.path.isfile(args.input):
        ap.error(f"输入 h5ad 不存在: {args.input}")

    if args._worker_size is not None:
        row = measure_in_fresh_worker(
            input_path=args.input,
            n_target=args._worker_size,
            epochs=args.epochs,
            memory_sample_interval_ms=args.memory_sample_interval_ms,
            seed=args.seed,
            load_only=args._worker_load_only,
        )
        print(RESULT_PREFIX + json.dumps(row, ensure_ascii=True), flush=True)
        return

    rows = []
    for n_target in args.sizes:
        print(f"启动 fresh worker: requested n={n_target}", flush=True)
        row = run_worker_subprocess(args, n_target)
        rows.append(row)
        # 每完成一个 worker 就落盘，长评测中途失败时保留已完成规模。
        write_results(rows, args.output)
        print_result(row)
    print(
        f"完成 -> {args.output}（每个规模均由独立 worker 测量 load+setup+fit）"
    )


if __name__ == "__main__":
    main()
