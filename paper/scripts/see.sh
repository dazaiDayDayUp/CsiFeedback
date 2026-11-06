#!/usr/bin/env bash
# 用法: see.sh <png路径> <问题>
set -uo pipefail
PNG="$1"; shift
Q="$*"
D="$(cd "$(dirname "$PNG")" && pwd)"
B="$(basename "$PNG")"
cd "$D"
/home/wzh/.kimi-code/bin/kimi -p "你是 IEEE 期刊论文的图形审稿人。请先阅读图片 $B，然后回答问题：$Q
要求：中文回答，先给出总体判断（可发表/需修改/不可用），再逐条列出具体问题及所在位置（左/右/上/下/中部），最后给可执行的修改建议。不超过 250 字。" -m kimi-code/k3.1-pro 2>/dev/null | grep -vE "^kimi version|^Warning:|^To resume|^$" | tail -12
