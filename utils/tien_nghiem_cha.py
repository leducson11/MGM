"""Tiên nghiệm từ cha cho Thompson sampling (fork m04a của nhóm KPDL, không có trong MGM gốc).

Mã gốc (``hgm.py``) dùng ``Beta(1 + s, 1 + f)`` cho mỗi nút, tức là coi nút con mới như chưa biết gì. Với ngân sách
nhỏ và thứ tự chấm bài cố định, kết quả của MỘT bài đầu tiên quyết định nút con được chú ý bao nhiêu: nút sai bài
đầu có ``Beta(1, 2)`` (trung bình 0,33) và, vì ``mean_utility = 0``, bị loại khỏi danh sách ứng viên làm cha. Nhóm
đo được bài đầu tiên (``javascript__queen-attack``) là nhiễu: chính agent ban đầu chỉ đúng 1/5 lần chấm lại.

Cải tiến: nút con là bản sửa nhỏ của nút cha, nên trước khi có dữ liệu ta tin nó giỏi gần bằng cha. Thêm ``m`` bài
giả theo tỉ lệ đúng ``p`` của cha::

    theta_i ~ Beta(1 + s_i + m * p_cha(i), 1 + f_i + m * (1 - p_cha(i)))

``m = 0`` (mặc định) cho đúng hành vi của tác giả. Nút gốc (agent ban đầu) không có tiên nghiệm.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

TienNghiem = tuple[float, float]
KHONG_CO: TienNghiem = (0.0, 0.0)


def ty_le_cha(node: Any, nodes: Mapping[int, Any]) -> Optional[float]:
    """Tỉ lệ đúng trên các lần chấm của chính nút cha; cha chưa được chấm thì lấy lên đời trên.

    Trả về ``None`` nếu nút là gốc hoặc không tổ tiên nào có lần chấm (hoặc đứt chuỗi cha).
    """
    da_qua = set()
    cha_id = getattr(node, "parent_id", None)
    while cha_id is not None and cha_id not in da_qua:
        da_qua.add(cha_id)
        cha = nodes.get(cha_id)
        if cha is None:
            return None
        do = list(cha.utility_measures)
        if do:
            return float(sum(do)) / len(do)
        cha_id = getattr(cha, "parent_id", None)
    return None


def gia_tri_tien_nghiem(node: Any, nodes: Mapping[int, Any], m: float) -> TienNghiem:
    """``(m * p, m * (1 - p))`` với ``p = ty_le_cha``; ``(0, 0)`` khi ``m <= 0`` hoặc không có ``p``."""
    if m <= 0:
        return KHONG_CO
    p = ty_le_cha(node, nodes)
    if p is None:
        return KHONG_CO
    return (m * p, m * (1.0 - p))


def trung_binh_co_tien_nghiem(utility_measures: Sequence[float], tien_nghiem: TienNghiem) -> float:
    """``(s + a0) / (n + a0 + b0)``; nút chưa chấm lần nào trả ``inf`` giống ``Node.mean_utility`` của mã gốc."""
    n = len(utility_measures)
    if n == 0:
        return math.inf
    a0, b0 = tien_nghiem
    return (float(sum(utility_measures)) + a0) / (n + a0 + b0)


def la_ung_vien_cha(node: Any, nodes: Mapping[int, Any], m: float) -> bool:
    """Điều kiện làm ứng viên cha trong ``expand()``.

    ``m <= 0``: đúng mã gốc (``mean_utility`` hữu hạn và > 0). ``m > 0``: đã được chấm ít nhất một lần và trung bình
    có tiên nghiệm > 0, nên nút 0/1 có cha giải được bài vẫn là ứng viên.
    """
    if m <= 0:
        mu = node.mean_utility
        return bool(math.isfinite(mu) and mu > 0)
    if len(node.utility_measures) == 0:
        return False
    return trung_binh_co_tien_nghiem(node.utility_measures, gia_tri_tien_nghiem(node, nodes, m)) > 0


def cong_tien_nghiem(alphas: Sequence[float], betas: Sequence[float],
                     tien_nghiem: Optional[Sequence[TienNghiem]]) -> tuple[list, list]:
    """Cộng tiên nghiệm vào tham số Beta; ``tien_nghiem=None`` trả lại nguyên ``alphas``, ``betas``."""
    if tien_nghiem is None:
        return alphas, betas
    if len(tien_nghiem) != len(alphas):
        raise ValueError("tien_nghiem phải cùng độ dài với evals")
    return ([a + t[0] for a, t in zip(alphas, tien_nghiem)],
            [b + t[1] for b, t in zip(betas, tien_nghiem)])
