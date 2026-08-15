# Ghi chép dự án

Cập nhật 2026-08-15. Tài liệu này ghi **đã làm gì, đã quyết gì, còn gì phải làm**.
Cách pipeline hoạt động xem `pipeline/NOTES.md`; lệnh chạy xem `pipeline/README.md`.

## Mục tiêu

Train model TTS đọc truyện bằng giọng tiếng Việt, giọng một người kể chuyện audiobook.
Bước hiện tại là chuẩn bị dataset.

## Dữ liệu nguồn

5 file trong `original/`, tổng **4.64 giờ**. Cả 5 đều là **AAC 128 kbps 44.1 kHz stereo**
dù đuôi file là `.mp3`.

| file | thời lượng |
| --- | --- |
| `sample.mp3` | 57.4 phút |
| `sample02.mp3` | 55.1 phút |
| `sample03.mp3` | 54.4 phút |
| `sample04.mp3` | 57.7 phút |
| `sample05.mp3` | 53.9 phút |

Nguồn 128 kbps là mức trần chất lượng của model. Đã cân nhắc và chấp nhận.

## Quyết định đã chốt

- **Train từ đầu**, không fine-tune từ checkpoint có sẵn
- **Một người đọc** cho tất cả file, dùng chung nhãn speaker `narrator`
- **Không lọc trùng** đoạn intro lặp giữa các tập — audio thế nào thì transcript thế đó
- Train sẽ chạy trên **GPU thuê**, không train trên MacBook

## Hệ quả của việc train từ đầu

Cần **10–25 giờ** audio dùng được, không phải 1–3 giờ như fine-tune. Với sản lượng ~90%,
4.64 giờ hiện có cho ra ~4.2 giờ, tức khoảng một phần tư mức tối thiểu.

| mục tiêu | audio thô | số file | cần thêm |
| --- | --- | --- | --- |
| 10 giờ | 11.1 giờ | 12 | 7 file |
| 15 giờ | 16.7 giờ | 18 | 13 file |
| 20–24 giờ | 22–27 giờ | 24–29 | 19–24 file |

Hai điều đặc thù của train-từ-đầu:

1. **Độ phủ âm tiết là ràng buộc quyết định.** Model không thừa hưởng cách phát âm từ base nào,
   nên âm tiết nào không có trong dataset thì lúc inference sẽ đọc sai. Một bộ truyện có vốn từ hạn
   chế, nên cần đếm số âm tiết riêng biệt và phân bố thanh điệu trong `dataset/metadata.csv`. Thêm
   nhiều giờ cùng một bộ truyện không giải quyết được vấn đề này.
2. **Phải tự xây front-end tiếng Việt.** Bert-VITS2 gắn cứng symbol set, g2p và cleaner theo từng
   ngôn ngữ, chỉ hỗ trợ ZH/JP/EN. Cần bộ âm vị tiếng Việt, xử lý 6 thanh, và nối PhoBERT
   (`vinai/phobert-base`). Đây là phần việc riêng, thường tốn hơn cả việc train.

Nếu không tìm được fork Bert-VITS2 có checkpoint tiếng Việt, đáng xem F5-TTS hoặc viXTTS — chúng có
base tiếng Việt sẵn, và `dataset/metadata.csv` dạng LJSpeech dùng được luôn cho cả hai.

## Bốn lỗi đã tìm ra và cách xử lý

Đây là lý do pipeline có các bước `align`, đọc lại từng clip, và kiểm tra độ đầy đủ.
Cả bốn lỗi đều do **nghe thử clip** mà phát hiện, không phải do bộ lọc tự bắt được.

**1. Cắt clip giữa âm tiết.** 34% ranh giới clip được cắt tại chỗ không có khoảng nghỉ nào, vì
word timestamp của Whisper không đáng tin và hàm cắt cũ rơi vào nhánh chia đôi số từ khi không tìm
được chỗ nghỉ. Sửa: lấy **khoảng lặng đo trực tiếp từ audio** làm nguồn chân lý cho điểm cắt, và
`boundary_report` kiểm tra lại sau khi cắt. Kết quả: 0 vi phạm.

**2. Từ bị gán sai phía chỗ nghỉ.** Text có từ mà audio không có, hoặc ngược lại. Đo được: 69%
khoảng nghỉ bị nằm lọt trong span của một từ, vì Whisper dùng DTW nên nuốt các chỗ nghỉ. Sửa: thêm
bước `align` dùng CTC forced alignment tính lại thời gian từng từ, giữ nguyên chữ của Whisper.
Độ sạch khoảng nghỉ: 23% → 69%.

**3. Whisper bịa nội dung.** Nó chèn câu kêu gọi đăng ký kênh YouTube thay cho nội dung truyện thật,
suốt gần một phút audio, và **tự tin vào câu bịa** (`asr_prob = 0.94`). Không thể dựa vào confidence
của Whisper. Sửa: chấm điểm CTC để phát hiện, rồi **đọc lại từng clip cô lập** — clip vài giây không
đủ chỗ cho Whisper trôi vào chuỗi nó học thuộc. Clip bịa nặng nhất đi từ 0.39 lên 0.93.

**4. Whisper bỏ sót chữ.** Điểm CTC alignment **mù với lỗi thiếu chữ**, vì nó chỉ đo các token được
đưa vào có khớp tín hiệu hay không; text là tập con của lời nói thì vẫn đạt điểm cao. Sửa: thêm chỉ
số độ đầy đủ (tỉ lệ độ dài text so với greedy CTC decode) làm tiêu chí kích hoạt đọc lại thứ hai.

## Trạng thái hiện tại

Chỉ `sample.mp3` đã xử lý, và **chỉ 5 phút đầu** (bản preview). 4 file còn lại chưa chạy.

- 51 clip trong `dataset/`, 4.5 phút, 90% sản lượng
- Mọi clip có `ctc_score` ≥ 0.795 và độ đầy đủ ≥ 0.947
- 0 biên cắt nằm trong tiếng nói

`work/sample/edit.txt` đã đổi tên thành `edit.txt.preview5min` vì nó ứng với bản preview; để nguyên
thì khi chạy full sẽ gán text sai vào clip khác.

## Việc tiếp theo

1. Chạy full 5 file: `pipeline/.venv/bin/python pipeline/run.py --force` — khoảng **2.8 giờ**,
   phần lớn là ASR
2. Nghe thử vài clip, kiểm hai dòng log: `align` phải có số sau lớn hơn số trước, và
   `segment: kiem tra OK`
3. Đếm độ phủ âm tiết trong `dataset/metadata.csv` để biết dữ liệu có đủ đa dạng hay không
4. Thu thêm ~13 file nữa để đạt 15 giờ
5. Xác định fork Bert-VITS2 có tiếng Việt, hoặc chuyển sang F5-TTS / viXTTS

## Giới hạn còn lại

Bộ lọc hiện tại bắt được lỗi **thô**: bịa nội dung, mất cả cụm từ, cắt giữa âm tiết. Nó **không** bắt
được sai một chữ đơn lẻ khi Whisper nghe lẫn hai từ gần âm — loại này vẫn cần tai người. Ở tỉ lệ vài
phần trăm rải rác thì nó hòa tan được khi train, vì model học trên toàn tập. Nhưng nếu một tên riêng
lặp lại xuyên suốt truyện mà luôn bị nghe sai thì model sẽ học sai hẳn cách đọc từ đó — chỗ đáng để
mắt nhất khi kiểm tra.
