"""Archive returned dpt_* PBS logs by task, preserving filenames and contents.

Run from any directory: python tools/archive_pbs_logs.py
Only files already returned to the repository root are moved. Future PBS output
paths are unchanged; rerun this tool after other jobs finish.
"""
import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / 'results/pbs_logs'
PATTERN = re.compile(r'^(dpt_[\w-]+)\.([oe])(\d+)$')


def category(name):
    if name == 'dpt_check':
        return '01_environment'
    if name.startswith('dpt_prep'):
        return '02_preprocessing'
    if name.startswith(('dpt_fold', 'dpt_neg', 'dpt_zero', 'dpt_pos', 'dpt_all')):
        frame = 'frame220' if name.endswith('220') else 'frame160' if name.endswith('160') else 'other'
        return f'03_training/{frame}'
    if name.startswith('dpt_err'):
        return '04_error_figures'
    if name.startswith('dpt_paper'):
        return '05_paper_figures'
    if name in ('dpt_audit', 'dpt_splits'):
        return '06_diagnostics'
    return '07_other'


def main():
    planned = []
    for src in sorted(REPO.glob('dpt_*')):
        match = PATTERN.fullmatch(src.name)
        if not src.is_file() or not match:
            continue
        dst = DEST / category(match[1]) / src.name
        if dst.exists():
            raise SystemExit(f'Refusing to overwrite {dst}; no files moved.')
        planned.append((src, dst, src.stat().st_size))
    for src, dst, size in planned:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        assert dst.stat().st_size == size
        print(f'{src.name} -> {dst.relative_to(REPO)}')
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / 'index.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['job_name', 'job_id', 'stream', 'bytes', 'path'])
        for path in sorted(DEST.rglob('dpt_*')):
            match = PATTERN.fullmatch(path.name)
            if match and path.is_file():
                writer.writerow([match[1], match[3], 'stdout' if match[2] == 'o' else 'stderr',
                                 path.stat().st_size, str(path.relative_to(REPO))])
    (DEST / 'README.md').write_text('''# PBS 日志归档\n\n保留原始文件名：`.o作业号` 是标准输出，`.e作业号` 是标准错误。错误流可能为空，也可能仅包含非致命警告。\n\n- `01_environment/`：CUDA 扩展编译与环境检查。\n- `02_preprocessing/`：160/220 帧预处理。\n- `03_training/frame160/`、`frame220/`：单折、配置比较或全部折训练，包括失败尝试。\n- `04_error_figures/`：误差着色图及比较汇总。\n- `05_paper_figures/`：Step 7 整套论文图和表。\n- `06_diagnostics/`：数据核查、训练/验证/测试重新评估。\n- `07_other/`：其他作业。\n\n完整索引见 `index.csv`。归档不修改 PBS 已提交作业的输出路径；尚未结束的作业以后仍可能在仓库根目录生成日志。完成后在仓库运行 `python tools/archive_pbs_logs.py` 可再次归档。不要在日志仍由程序写入时手动移动它。此目录位于已被 Git 忽略的 results 下。\n''')
    print(f'Archived {len(planned)} files; index: {DEST / "index.csv"}')


if __name__ == '__main__':
    main()
