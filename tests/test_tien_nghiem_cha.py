"""Test utils/tien_nghiem_cha.py (fork m04a): tiên nghiệm từ cha cho Thompson sampling. Không gọi LLM, không cần Docker."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chon_cha import chon_cha_co_toan_tu  # noqa: E402
from utils.tien_nghiem_cha import (KHONG_CO, cong_tien_nghiem, gia_tri_tien_nghiem,  # noqa: E402
                                   la_ung_vien_cha, trung_binh_co_tien_nghiem, ty_le_cha)


class NutGia:
    """Giống các trường tree.Node dùng tới (không import tree.py để khỏi kéo theo hgm_utils)."""

    def __init__(self, id, parent_id, utility_measures, commit_id=None):
        self.id = id
        self.parent_id = parent_id
        self.utility_measures = list(utility_measures)
        self.commit_id = commit_id or f"n{id}"

    @property
    def num_evals(self):
        return len(self.utility_measures)

    @property
    def mean_utility(self):  # đúng như tree.Node
        if self.num_evals == 0:
            return np.inf
        return np.sum(self.utility_measures) / self.num_evals


def cay_hgmx_sau_luot_1():
    """Trạng thái HGMx r1 sau lượt 1: agent ban đầu 36/59, nút 1 sai bài đầu (0/1)."""
    goc = NutGia(0, None, [1] * 36 + [0] * 23, "initial")
    n1 = NutGia(1, 0, [0])
    return {0: goc, 1: n1}


def ts_goc(evals):
    """TS_sample của hgm.py (mã gốc, cool_down = False)."""
    alphas = [1 + np.sum(de) for de in evals]
    betas = [1 + len(de) - np.sum(de) for de in evals]
    return np.argmax(np.random.beta(alphas, betas))


def ts_fork(evals, tien_nghiem=None):
    """TS_sample của fork (cool_down = False)."""
    alphas = [1 + np.sum(de) for de in evals]
    betas = [1 + len(de) - np.sum(de) for de in evals]
    alphas, betas = cong_tien_nghiem(alphas, betas, tien_nghiem)
    return np.argmax(np.random.beta(alphas, betas))


# ---- ty_le_cha / gia_tri_tien_nghiem ----

def test_ty_le_cha_la_ti_le_dung_cua_chinh_nut_cha():
    nodes = cay_hgmx_sau_luot_1()
    assert ty_le_cha(nodes[1], nodes) == pytest.approx(36 / 59)


def test_nut_goc_khong_co_tien_nghiem():
    nodes = cay_hgmx_sau_luot_1()
    assert ty_le_cha(nodes[0], nodes) is None
    assert gia_tri_tien_nghiem(nodes[0], nodes, 4) == KHONG_CO


def test_cha_chua_cham_thi_lay_len_doi_tren():
    nodes = cay_hgmx_sau_luot_1()
    nodes[2] = NutGia(2, 1, [])           # cha chưa chấm lần nào
    nodes[1].utility_measures = []
    nodes[3] = NutGia(3, 2, [1])
    assert ty_le_cha(nodes[3], nodes) == pytest.approx(36 / 59)


def test_dut_chuoi_cha_tra_none():
    nodes = {5: NutGia(5, 99, [1])}
    assert ty_le_cha(nodes[5], nodes) is None
    assert gia_tri_tien_nghiem(nodes[5], nodes, 4) == KHONG_CO


def test_cong_thuc_tien_nghiem():
    nodes = cay_hgmx_sau_luot_1()
    a0, b0 = gia_tri_tien_nghiem(nodes[1], nodes, 4)
    assert a0 == pytest.approx(4 * 36 / 59) and b0 == pytest.approx(4 * 23 / 59)
    assert a0 + b0 == pytest.approx(4)


def test_m_bang_0_khong_co_tien_nghiem():
    nodes = cay_hgmx_sau_luot_1()
    assert gia_tri_tien_nghiem(nodes[1], nodes, 0) == KHONG_CO


def test_trung_binh_co_tien_nghiem():
    # nút 0/1, cha 36/59, m = 4: (0 + 2,44) / (1 + 4) ≈ 0,49
    assert trung_binh_co_tien_nghiem([0], (4 * 36 / 59, 4 * 23 / 59)) == pytest.approx(4 * 36 / 59 / 5)
    assert trung_binh_co_tien_nghiem([], (1.0, 1.0)) == math.inf
    assert trung_binh_co_tien_nghiem([1, 0], KHONG_CO) == pytest.approx(0.5)


# ---- la_ung_vien_cha ----

@pytest.mark.parametrize("do", [[], [0], [0, 0], [1], [1, 0], [0, 1, 1]])
def test_m_bang_0_dung_dieu_kien_goc(do):
    nodes = cay_hgmx_sau_luot_1()
    nodes[1].utility_measures = list(do)
    n = nodes[1]
    goc = bool(np.isfinite(n.mean_utility) and n.mean_utility > 0)
    assert la_ung_vien_cha(n, nodes, 0) == goc


def test_m_duong_giu_nut_0_tren_1_lam_ung_vien():
    nodes = cay_hgmx_sau_luot_1()
    assert not la_ung_vien_cha(nodes[1], nodes, 0)      # mã gốc: bị loại
    assert la_ung_vien_cha(nodes[1], nodes, 4)          # fork: vẫn là ứng viên
    assert la_ung_vien_cha(nodes[0], nodes, 4)


def test_m_duong_van_loai_nut_chua_cham():
    nodes = cay_hgmx_sau_luot_1()
    nodes[2] = NutGia(2, 0, [])
    assert not la_ung_vien_cha(nodes[2], nodes, 4)
    assert not la_ung_vien_cha(nodes[2], nodes, 0)


def test_m_duong_loai_nut_khi_cha_cung_0_phan_tram():
    nodes = {0: NutGia(0, None, [0, 0]), 1: NutGia(1, 0, [0])}
    assert not la_ung_vien_cha(nodes[1], nodes, 4)


# ---- TS_sample với tiên nghiệm ----

def test_cong_tien_nghiem_none_tra_nguyen():
    a, b = [1, 2], [3, 4]
    assert cong_tien_nghiem(a, b, None) == (a, b)


def test_cong_tien_nghiem_sai_do_dai_bao_loi():
    with pytest.raises(ValueError):
        cong_tien_nghiem([1, 2], [1, 2], [(0, 0)])


@pytest.mark.parametrize("seed", range(20))
def test_khong_tien_nghiem_cung_lua_chon_va_cung_trang_thai_ngau_nhien_nhu_ma_goc(seed):
    evals = [[1] * 36 + [0] * 23, [0], [1, 0, 1], [1]]
    np.random.seed(seed)
    goc = ts_goc(evals)
    sau_goc = np.random.random()
    np.random.seed(seed)
    fork = ts_fork(evals, None)
    assert fork == goc and np.random.random() == sau_goc


@pytest.mark.parametrize("seed", range(20))
def test_tien_nghiem_bang_0_qua_chon_cha_co_toan_tu_giong_goi_thang(seed):
    """Nhánh m > 0 của expand() bọc (evals, tiên nghiệm) cho chon_cha_co_toan_tu; với tiên nghiệm 0 phải ra cùng nút."""
    evals = [[1] * 36 + [0] * 23, [0], [1, 0, 1]]
    nodes = ["goc", "n1", "n2"]
    dk = lambda n: ([(1.0, "A", {})], None)  # noqa: E731
    np.random.seed(seed)
    thang, _, _ = chon_cha_co_toan_tu(nodes, evals, dk, ts_goc)
    np.random.seed(seed)
    boc, _, _ = chon_cha_co_toan_tu(nodes, list(zip(evals, [KHONG_CO] * 3)), dk,
                                    lambda ds: ts_fork([d[0] for d in ds], [d[1] for d in ds]))
    assert boc == thang


def test_tien_nghiem_lam_nut_0_tren_1_duoc_chon_nhieu_hon():
    """Bước đánh giá: nút 0/1 so với nút 1/1 (cùng cha 36/59). Mã gốc gần như bỏ nút 0/1; có tiên nghiệm thì không."""
    nodes = {0: NutGia(0, None, [1] * 36 + [0] * 23), 1: NutGia(1, 0, [0]), 2: NutGia(2, 0, [1])}
    evals = [nodes[1].utility_measures, nodes[2].utility_measures]
    tn = [gia_tri_tien_nghiem(nodes[i], nodes, 4) for i in (1, 2)]
    np.random.seed(0)
    goc = np.mean([ts_fork(evals, None) == 0 for _ in range(4000)])
    co_tn = np.mean([ts_fork(evals, tn) == 0 for _ in range(4000)])
    assert goc == pytest.approx(1 / 6, abs=0.03)      # P(Beta(1,2) > Beta(2,1)) = 1/6
    # P(Beta(3,44; 3,56) > Beta(4,44; 2,56)) ≈ 0,28 (mô phỏng 200 000 lần)
    assert co_tn == pytest.approx(0.28, abs=0.03)
