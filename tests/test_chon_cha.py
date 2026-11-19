"""Test utils/chon_cha.py (fork m04a): chọn nút cha, có tuỳ chọn "chọn cha khác". Không gọi LLM, không cần Docker."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chon_cha import chon_cha_co_toan_tu  # noqa: E402

CO = ([(1.0, "A", {})], "ngu-canh")     # có toán tử dùng được
KHONG = ([], "ngu-canh-rong")           # "No eligible"


class TsGhiLai:
    """ts_sample giả: trả về lần lượt các chỉ số cho trước, ghi lại danh sách được đưa vào."""

    def __init__(self, chi_so):
        self.chi_so = list(chi_so)
        self.goi = []

    def __call__(self, evals):
        self.goi.append(list(evals))
        return self.chi_so.pop(0)


def ts_nhu_hgm(evals):
    """Giống TS_sample của hgm.py khi cool_down = False."""
    alphas = [1 + np.sum(de) for de in evals]
    betas = [1 + len(de) - np.sum(de) for de in evals]
    return np.argmax(np.random.beta(alphas, betas))


def test_co_tat_boc_mot_lan_ke_ca_khi_khong_du_dieu_kien():
    ts = TsGhiLai([1])
    nut, kq, bo = chon_cha_co_toan_tu(["goc", "nut1"], ["e0", "e1"], lambda n: KHONG, ts, chon_cha_khac=False)
    assert nut == "nut1" and kq == KHONG and bo == []
    assert ts.goi == [["e0", "e1"]]                     # đúng một lần bốc, như mã gốc


def test_co_bat_bo_qua_nut_khong_du_dieu_kien_va_boc_lai():
    """Tình huống HGM r1 lúc 20:56: bốc trúng nút 1 (chưa sai bài nào), nút gốc có bài sai nên A dùng được."""
    dk = {"goc": CO, "nut1": KHONG}
    ts = TsGhiLai([1, 0])
    nut, kq, bo = chon_cha_co_toan_tu(["goc", "nut1"], ["e0", "e1"], dk.__getitem__, ts, chon_cha_khac=True)
    assert nut == "goc" and kq == CO and bo == ["nut1"]
    assert ts.goi == [["e0", "e1"], ["e0"]]             # lần 2 chỉ bốc trong các nút còn lại


def test_co_bat_nhung_nut_dau_du_dieu_kien_thi_nhu_co_tat():
    ts = TsGhiLai([0])
    nut, kq, bo = chon_cha_co_toan_tu(["a", "b"], ["ea", "eb"], lambda n: CO, ts, chon_cha_khac=True)
    assert nut == "a" and kq == CO and bo == [] and len(ts.goi) == 1


def test_moi_nut_deu_khong_du_dieu_kien_tra_ve_nut_dau_tien():
    """Hết ứng viên: giữ hành vi gốc (expand ghi "No eligible" cho nút bốc đầu tiên rồi quay về đánh giá)."""
    ts = TsGhiLai([2, 0, 0])
    nut, kq, bo = chon_cha_co_toan_tu(["a", "b", "c"], [1, 2, 3], lambda n: KHONG, ts, chon_cha_khac=True)
    assert nut == "c" and kq == KHONG and bo == ["c", "a", "b"]
    assert ts.goi == [[1, 2, 3], [1, 2], [2]]


def test_mot_ung_vien():
    nut, kq, bo = chon_cha_co_toan_tu(["a"], ["ea"], lambda n: CO, TsGhiLai([0]), chon_cha_khac=True)
    assert (nut, kq, bo) == ("a", CO, [])
    nut, kq, bo = chon_cha_co_toan_tu(["a"], ["ea"], lambda n: KHONG, TsGhiLai([0]), chon_cha_khac=True)
    assert (nut, kq, bo) == ("a", KHONG, ["a"])


def test_khong_co_ung_vien():
    assert chon_cha_co_toan_tu([], [], lambda n: CO, TsGhiLai([]), chon_cha_khac=True) == (None, None, [])


def test_do_dai_khac_nhau_thi_bao_loi():
    with pytest.raises(ValueError):
        chon_cha_co_toan_tu(["a"], [], lambda n: CO, TsGhiLai([0]))


@pytest.mark.parametrize("seed", range(20))
def test_co_tat_cho_cung_lua_chon_va_cung_trang_thai_ngau_nhien_nhu_ma_goc(seed):
    """Cờ tắt: cùng nút với ``nodes[TS_sample(decendant_evals)]`` của mã gốc và tiêu thụ đúng số lần bộ sinh
    ngẫu nhiên, nên các lần bốc sau trong lần chạy cũng không đổi."""
    evals = [[1] * 36 + [0] * 23, [1], [1, 0, 1], [0, 0]]
    nodes = ["goc", "n1", "n2", "n3"]
    np.random.seed(seed)
    goc = nodes[ts_nhu_hgm(evals)]
    sau_goc = np.random.random()
    np.random.seed(seed)
    nut, _, _ = chon_cha_co_toan_tu(nodes, evals, lambda n: KHONG, ts_nhu_hgm, chon_cha_khac=False)
    assert nut == goc and np.random.random() == sau_goc


def test_co_bat_khong_bao_gio_chon_nut_khong_du_dieu_kien_khi_con_nut_khac():
    evals = [[1] * 36 + [0] * 23, [1] * 5]
    dk = {"goc": CO, "n1": KHONG}
    for seed in range(50):
        np.random.seed(seed)
        nut, kq, _ = chon_cha_co_toan_tu(["goc", "n1"], evals, dk.__getitem__, ts_nhu_hgm, chon_cha_khac=True)
        assert nut == "goc" and kq == CO
