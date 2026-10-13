# -*- coding: utf-8 -*-
"""
判定表 2「固定系数白化」列实现的是 ZCA（对称）白化还是 PCA 白化。

判定标准（两者都使协方差单位化，差别在是否保留原方向）：
    ZCA:  W = Σ^{-1/2} = V Λ^{-1/2} Vᵀ      —— V 在两侧，方向不变
    PCA:  W = Λ^{-1/2} Vᵀ                    —— V 只在右侧，旋转到主成分基

可检验的判别式：
    1) W 是否对称？      ZCA 对称；PCA 一般不对称
    2) W @ Wᵀ 是否等于 Σ^{-1}（而非仅 Λ^{-1}）？
    3) 对同一份数据，W 与 Λ^{-1/2}Vᵀ 是否相等？
"""
from __future__ import annotations

import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
os.chdir(TOOLS)

import calibration as CAL                                       # noqa: E402


def main():
    rng = np.random.RandomState(0)
    A = rng.randn(4, 4) + np.eye(4) * 2.0
    X = rng.randn(400, 4) @ A + rng.randn(400, 4) * 0.05
    y = rng.randint(0, 5, 400)

    # 走真实代码路径：与 apply_unconditional_whitening.py 完全一致
    W = CAL.fit("pca", X, y, 0.1)

    # 解析解
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / (len(X) - 1)
    d = S.shape[0]
    Ss = (1 - 0.1) * S + 0.1 * np.trace(S) / d * np.eye(d)      # 与 _inv_sqrt 的收缩一致
    w, V = np.linalg.eigh(Ss)
    W_zca = V @ np.diag(1 / np.sqrt(w)) @ V.T
    W_pca = np.diag(1 / np.sqrt(w)) @ V.T

    print("=" * 74)
    print("判定表 2「固定系数白化」列的实现")
    print("=" * 74)
    print("  代码路径: CAL.fit(\"pca\", f, y, 0.1)   （apply_unconditional_whitening.py）")
    print()
    print("  判别式 1  W 是否对称")
    print("     max|W − Wᵀ| = %.3e   → %s" % (np.abs(W - W.T).max(),
                                            "对称（ZCA 特征）" if np.allclose(W, W.T) else "不对称"))
    print()
    print("  判别式 2  W 与 Σ^{-1/2}（ZCA）是否相等")
    print("     max|W − W_zca| = %.3e   → %s" % (np.abs(W - W_zca).max(),
                                               "相等" if np.allclose(W, W_zca) else "不等"))
    print()
    print("  判别式 3  W 与 Λ^{-1/2}Vᵀ（PCA）是否相等")
    print("     max|W − W_pca| = %.3e   → %s" % (np.abs(W - W_pca).max(),
                                               "相等" if np.allclose(W, W_pca) else "不等"))
    print()
    print("  判别式 4  两者是否本就不同")
    print("     max|W_zca − W_pca| = %.3e   → %s" % (np.abs(W_zca - W_pca).max(),
                                                   "不同" if not np.allclose(W_zca, W_pca) else "相同"))
    print()
    # 在余弦相似度下游的差异
    Zz = CAL.l2((X - X.mean(0)) @ W_zca)
    Zp = CAL.l2((X - X.mean(0)) @ W_pca)
    sz = (Zz @ Zz.T)[np.triu_indices(len(X), 1)]
    sp = (Zp @ Zp.T)[np.triu_indices(len(X), 1)]
    print("  余弦相似度下游差异（本文协议用余弦判别）")
    print("     逐对相似度最大差异 = %.4f" % np.abs(sz - sp).max())
    print()
    print("=" * 74)
    verdict = ("ZCA（对称）白化" if (np.allclose(W, W.T) and np.allclose(W, W_zca)
                                     and not np.allclose(W, W_pca)) else "需人工判断")
    print("  判定：表 2「固定系数白化」列实现的是  %s" % verdict)
    print("=" * 74)


if __name__ == "__main__":
    main()
