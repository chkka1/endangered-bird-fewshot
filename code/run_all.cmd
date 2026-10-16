@echo off
REM ============================================================================
REM  核验包一键核验
REM
REM  本包为核验包，随包提供 code/ dataset/ results/ paper/ 与许可证文件。
REM  未能随包发布的资产（体积大或受第三方许可限制）：
REM      _tools/feat_cache/  冻结特征缓存
REM      data/               CUB / NABirds 原始图像
REM      cub_parsed/         划分与解析中间件
REM      checkpoints/        主干权重
REM  依赖这些资产的脚本会打印说明并跳过（退出码 0），不代表论文数据有问题。
REM
REM  用法：把下一行的 python 换成你的解释器路径（或确保它在 PATH 中），
REM        然后双击本文件或在终端运行。全部退出码 0 即表示稿件与结果一致。
REM ============================================================================
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0;%~dp001_data;%~dp002_cache;%~dp003_train;%~dp004_eval;%~dp005_verify;%~dp006_paper;%PYTHONPATH%

set PY=python
cd /d %~dp0

for %%f in (verify_prose_aggregates check_causal_drops audit_endb_manifest audit_all_data audit_all_data2 audit_all_data3 audit_all_data4) do (
  echo ============================================================
  echo  %%f
  echo ============================================================
  "%PY%" "%~dp005_verify\%%f.py"
)

echo.
echo Done.
pause
