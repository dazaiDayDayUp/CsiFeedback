"""Pytest 本地插件：在每个测试模块完成后打印一句总结。"""

import io
import sys
from pathlib import Path


# 记录每个模块的测试状态
_module_results = {}


def _utf8_stdout():
    """将 stdout 重新包装为 UTF-8，确保中文和 emoji 能正确打印。"""
    if isinstance(sys.stdout, io.TextIOWrapper):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            try:
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace"
                )
            except Exception:
                pass


def pytest_sessionstart(session):
    """测试会话开始时配置 UTF-8 标准输出。"""
    _utf8_stdout()


def pytest_runtest_logreport(report):
    """收集每个模块的通过/失败数量。"""
    if report.when != "call":
        return
    module_path = report.nodeid.split("::")[0]
    _module_results.setdefault(module_path, {"passed": 0, "failed": 0})
    if report.outcome == "passed":
        _module_results[module_path]["passed"] += 1
    elif report.outcome == "failed":
        _module_results[module_path]["failed"] += 1


def pytest_sessionfinish(session, exitstatus):
    """测试会话结束时，按模块打印简洁总结。"""
    _utf8_stdout()
    print()
    print("=" * 60)
    print("测试模块总结")
    print("=" * 60)
    for module_path, counts in _module_results.items():
        file_path = Path(module_path)
        try:
            module_doc = _extract_module_doc(file_path)
        except Exception:
            module_doc = ""
        status = "✅ 通过" if counts["failed"] == 0 else "❌ 失败"
        summary = module_doc or f"{file_path.stem} 测试"
        print(f"{status} | {file_path.name:<22} | {summary}")
    print("=" * 60)


def _extract_module_doc(path: Path) -> str:
    """提取测试文件模块级 docstring 的第一行。"""
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    if '"""' in source:
        start = source.find('"""') + 3
        end = source.find('"""', start)
        doc = source[start:end].strip()
        first_line = doc.splitlines()[0] if doc else ""
        return first_line.strip()
    return ""
