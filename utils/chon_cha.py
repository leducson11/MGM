"""Chọn nút cha khi mở rộng cây, có tuỳ chọn "chọn cha khác" (fork m04a của nhóm KPDL, không có trong MGM gốc).

Mã gốc (``hgm.py``, ``expand()``) bốc MỘT nút cha bằng Thompson sampling. Nếu nút đó không có toán tử tự sửa nào
dùng được (A cần bài sai, B cần đủ số lần đánh giá và hai bài sai, C cần nút ngang hàng có kết quả khác trên bài
chung), ``expand()`` ghi "No eligible self-improvement strategy" rồi trả False, và vòng lặp quay về ĐÁNH GIÁ thêm
một bài. Trong lần chạy HGM r1 của nhóm, việc này xảy ra 7 lần liền (7/20 lần đánh giá), trong khi agent ban đầu
(23 bài sai, toán tử A dùng được) cũng nằm trong danh sách ứng viên.

``chon_cha_co_toan_tu`` với ``chon_cha_khac=True``: loại nút không có toán tử dùng được rồi bốc lại bằng chính
Thompson sampling trên các ứng viên còn lại, tức là Thompson sampling có điều kiện trên tập nút có ít nhất một
toán tử dùng được. Với ``chon_cha_khac=False`` hàm bốc đúng một lần như mã gốc (cùng số lần gọi bộ sinh ngẫu
nhiên), nên cờ tắt cho đúng hành vi của tác giả.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


def chon_cha_co_toan_tu(
    ung_vien: Sequence[Any],
    danh_gia: Sequence[Any],
    du_dieu_kien: Callable[[Any], tuple[list, Any]],
    ts_sample: Callable[[list], int],
    chon_cha_khac: bool = False,
) -> tuple[Any, tuple[list, Any] | None, list]:
    """Bốc nút cha bằng ``ts_sample``; trả về ``(nút, kết quả điều kiện, các nút đã bỏ qua)``.

    ``danh_gia[i]`` là dữ liệu đưa cho ``ts_sample`` của ``ung_vien[i]`` (trong ``hgm.py``: các lần đánh giá
    của cả nhánh con). ``du_dieu_kien(nút)`` trả về ``(weighted, ngữ cảnh)``; ``weighted`` rỗng nghĩa là không
    toán tử nào dùng được (đúng điều kiện "No eligible" của mã gốc).

    - Cờ tắt: bốc một lần, trả về nút đó cùng kết quả điều kiện (``weighted`` có thể rỗng); danh sách bỏ qua rỗng.
    - Cờ bật: nút có ``weighted`` rỗng bị bỏ qua và bốc lại trên các ứng viên còn lại. Nếu mọi ứng viên đều không
      dùng được, trả về nút được bốc ĐẦU TIÊN cùng kết quả điều kiện của nó (rỗng), để ``expand()`` ghi
      "No eligible" và quay về đánh giá như mã gốc.
    - Không có ứng viên: trả về ``(None, None, [])``.
    """
    if len(ung_vien) != len(danh_gia):
        raise ValueError("ung_vien và danh_gia phải cùng độ dài")
    con_lai = list(range(len(ung_vien)))
    bo_qua: list = []
    dau_tien: tuple[Any, tuple[list, Any]] | None = None
    while con_lai:
        k = int(ts_sample([danh_gia[i] for i in con_lai]))
        i = con_lai[k]
        kq = du_dieu_kien(ung_vien[i])
        if dau_tien is None:
            dau_tien = (ung_vien[i], kq)
        if kq[0] or not chon_cha_khac:
            return ung_vien[i], kq, bo_qua
        bo_qua.append(ung_vien[i])
        con_lai.pop(k)
    if dau_tien is None:
        return None, None, []
    return dau_tien[0], dau_tien[1], bo_qua
