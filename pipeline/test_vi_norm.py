from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vi_norm

CASES = [
    ("Năm 1975 có 105 người.", "Năm một nghìn chín trăm bảy mươi lăm có một trăm lẻ năm người."),
    ("Giá 1.500.000 đồng.", "Giá một triệu năm trăm nghìn đồng."),
    ("Nhiệt độ 35°C và độ ẩm 80%.", "Nhiệt độ ba mươi lăm độ xê và độ ẩm tám mươi phần trăm."),
    ("Lúc 14:30 ngày 12/3/2024.", "Lúc mười bốn giờ ba mươi phút ngày mười hai tháng ba năm hai nghìn không trăm hai mươi tư."),
    ("Chương IV nói về 21 điều.", "Chương bốn nói về hai mươi mốt điều."),
    ("Anh ấy đi 25 km trong 1,5 giờ.", "Anh ấy đi hai mươi lăm ki lô mét trong một phẩy năm giờ."),
    ("Xếp thứ 4 và thứ 1.", "Xếp thứ tư và thứ nhất."),
    ("Số 0912345678 gọi lúc 7h30.", "Số không chín một hai ba bốn năm sáu bảy tám gọi lúc bảy giờ ba mươi phút."),
    ("Chỉ 1/2 số học sinh…", "Chỉ một phần hai số học sinh…"),
    ("TP.HCM có 9 triệu dân.", "thành phố Hồ Chí Minh có chín triệu dân."),
    ("Cà—phê & bánh ngọt", "Cà phê và bánh ngọt."),
    ("Trang 10-15 rất hay!!!", "Trang mười đến mười lăm rất hay!"),
    ("Anh ta nói 1.000 lần rồi", "Anh ta nói một nghìn lần rồi."),
    ("Còn 1005 đồng thôi.", "Còn một nghìn không trăm lẻ năm đồng thôi."),
]


def main() -> int:
    normalizer = vi_norm.build({"four_after_ten": "tư"})
    failures = 0
    for source, expected in CASES:
        got = normalizer(source)
        ok = got == expected
        failures += 0 if ok else 1
        print(("PASS " if ok else "FAIL ") + repr(source))
        if not ok:
            print(f"      mong doi: {expected}")
            print(f"      nhan duoc: {got}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
