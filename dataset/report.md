# Dataset vi_story

## Nguồn (1 file)

| id | file | kbps | gốc (phút) | đã xử lý (phút) | clip | dùng được (phút) |
| --- | --- | --- | --- | --- | --- | --- |
| `sample` | sample.mp3 | 129 | 57.45 | 57.45 | 577 | 52.4 |

## Kết quả
- Clip hợp lệ: **577**
- Tổng thời lượng: **0.873 giờ** (3144.07 s)
- Thời lượng trung bình: 5.449 s
- Phân bố thời lượng: {'min': 1.52, 'p10': 2.35, 'median': 6.145, 'p90': 7.7, 'max': 11.68}
- Phân bố số ký tự: {'min': 19.0, 'p10': 36.0, 'median': 84.0, 'p90': 112.0, 'max': 168.0}
- Train / Val: 566 / 11

## Lọc chất lượng
- Tổng segment: 587
- Nhận: 577 | Loại: 10

| Lý do loại | Số lượng |
| --- | --- |
| duration_too_short | 5 |
| duration_too_long | 4 |
| low_asr_confidence | 1 |

## Thời gian xử lý (giây)

| Bước | Giây |
| --- | --- |
| sample.decode | 7.6 |
| sample.vad_raw | 15.2 |
| sample.denoise | 34.8 |
| sample.vad_clean | 13.6 |
| sample.asr | 590.3 |
| sample.align | 67.9 |
| sample.segment | 0.2 |
| sample.cut | 5.7 |
| sample.text | 241.6 |
| sample.qc | 0.0 |
