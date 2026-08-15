# Ghi chú kỹ thuật



Lệnh dùng hằng ngày ở [README.md](README.md). File này giải thích vì sao pipeline làm như vậy.



## Các bước

| Bước | Việc làm | Kết quả trong `work/` |
| --- | --- | --- |
| `decode` | ffmpeg → mono 44.1 kHz float, high-pass 60 Hz, bản 16 kHz cho VAD/ASR | `raw_44k.wav`, `raw_16k.wav`, `source.json` |
| `vad` (raw) | Silero VAD trên bản gốc để tìm đoạn im lặng làm noise profile | `vad_raw.json` |
| `denoise` | Khử nhiễu theo backend đã chọn + chuẩn hóa độ to LUFS toàn file | `clean_44k.wav`, `clean_16k.wav` |
| `vad` (clean) | Silero VAD trên bản đã sạch → mốc ranh giới câu | `vad_clean.json` |
| `asr` | Whisper large-v3 tiếng Việt, lấy **chữ** (timestamp của nó không dùng để cắt) | `asr.json` |
| `align` | CTC forced alignment tính lại mốc thời gian từng từ | `aligned.json`, `align_stats.json` |
| `segment` | Đo khoảng lặng thật từ audio, chọn điểm cắt hợp lệ, gom thành clip, tự kiểm tra lại biên | `segments.json`, `segment_stats.json` |
| `cut` | Cắt clip từ bản 44.1 kHz, trim đầu/cuối, đo SNR / clipping / dBFS | `clips/*.wav`, `clips.json` |
| `text` | Đọc lại clip điểm thấp, giữ bản khớp audio hơn, rồi chuẩn hóa text | `texts.json`, `rescore_stats.json` |
| `qc` | Lọc theo thời lượng, tốc độ đọc, SNR, độ tự tin ASR, trùng lặp | `qc.json` |
| `export` | Xuất layout Bert-VITS2 + train/val + báo cáo | `dataset/` |

### Vì sao cần bước `align`

Word timestamp của Whisper sinh từ DTW trên cross-attention và **nuốt các chỗ nghỉ**: đo trên 5 phút
đầu của file mẫu, 69% khoảng nghỉ trong lời nói bị nằm lọt hoàn toàn trong span của một từ. Hệ quả là
từ bị gán sai phía của chỗ nghỉ — text có từ mà audio không có, hoặc ngược lại.

Bước `align` giữ nguyên **chữ** của Whisper nhưng tính lại **thời gian** bằng CTC forced alignment
(`torchaudio.functional.forced_align`) với model CTC tiếng Việt. CTC khớp trực tiếp chuỗi ký tự với
tín hiệu nên span của từ bó sát phần phát âm thật.

Audio được chia chunk tại các khoảng lặng dài, mỗi slice lấy thêm lề `chunk_margin` (2.5s) rộng hơn
sai lệch tối đa của Whisper (~1.9s), nên dù một từ bị gán lệch chunk thì âm thanh của nó vẫn nằm
trong slice và CTC vẫn định vị đúng.

Log của bước này in `silence_purity` trước và sau — tỉ lệ khoảng nghỉ **không** bị span của từ phủ
lên. Trên file mẫu: 23% → 69%. Nếu con số sau không cao hơn trước, model CTC không khớp ngôn ngữ
của audio.

### Chống Whisper bịa nội dung

Whisper chạy trên cả file sẽ decode theo cửa sổ 30 giây và có thể **trôi vào những câu nó học thuộc
từ phụ đề YouTube** — kiểu "hãy subscribe cho kênh…" — rồi thay thế hẳn nội dung thật. Nó tự tin vào
câu bịa đó: trên file mẫu, một clip bịa hoàn toàn vẫn có `asr_prob = 0.94`. Không thể dựa vào
confidence của Whisper để phát hiện.

Điểm CTC alignment thì phát hiện được, vì nó đo trực tiếp text có khớp tín hiệu hay không.
Đo trên file mẫu: clip bịa 0.27–0.59, clip đúng 0.63–0.99.

Bước `text` dùng điểm này để **sửa** chứ không chỉ loại:

1. Clip nào có `ctc_score < rescore.threshold` thì đọc lại bằng Whisper trên **đúng clip đó**, cô lập.
   Clip vài giây không đủ chỗ cho Whisper trôi vào chuỗi học thuộc.
2. Chấm điểm CTC cho cả text cũ và text mới trên cùng đoạn audio.
3. Giữ bản điểm cao hơn, chỉ khi hơn ít nhất `rescore.min_gain`.

Trên file mẫu, 17/18 clip được thay text, và clip bịa nặng nhất đi từ 0.39 lên 0.93. Sản lượng dùng
được tăng từ 56% lên 90% vì đây là **sửa** text sai, không phải loại bỏ audio tốt.

`qc.ctc_overrides_no_speech` tồn tại vì lý do liên quan: `no_speech_prob` là tín hiệu của Whisper
lượt 1, nên sau khi text đã được thay và xác minh bằng CTC thì nó đã lỗi thời. Clip có
`ctc_score` vượt ngưỡng này sẽ không bị loại vì `no_speech` nữa.

Đặt `rescore.threshold: 1.01` để đọc lại **toàn bộ** clip thay vì chỉ clip điểm thấp — chậm hơn
nhiều và đánh đổi: clip cô lập làm Whisper mất ngữ cảnh nên tên riêng và dấu câu có thể kém hơn.

### Cách chọn điểm cắt

Word timestamp của Whisper không đủ tin để đặt ranh giới clip, nên bước `segment` lấy **khoảng lặng
đo trực tiếp từ audio** làm nguồn chân lý:

1. Tính mức RMS từng frame 10ms, lấy median trong vùng VAD làm mức giọng nói tham chiếu,
   ngưỡng im lặng = mức đó trừ `silence_offset_db`.
2. Mỗi dải liên tục dưới ngưỡng dài hơn `min_cut_silence` là một điểm cắt ứng viên.
3. Một khoảng lặng **bị loại** nếu có từ vừa phủ nó ≥ `max_word_silence_overlap`, vừa có tốc độ
   giây/ký tự ≥ `max_word_rate_factor` lần median của cả file — dấu hiệu timestamp bị kéo giãn và
   không biết từ đó thuộc bên nào. Khi đó hai cụm được giữ trong cùng một clip nên text và audio
   không lệch nhau.
4. Các block giữa hai điểm cắt được gom lại tới `target_duration`, ưu tiên dừng ở chỗ có dấu kết câu.
5. Sau khi tạo clip, `boundary_report` đo lại mức âm thanh tại từng biên và báo động nếu có biên nào
   nằm trong vùng có tiếng nói. Log của bước này phải hiện `kiem tra OK`.

Ngưỡng ở bước 3 là tương đối theo median nên tự thích ứng với tốc độ đọc, không phải hằng số cứng.

## Chuẩn hóa text

`vi_norm.py` xử lý thuần Python, không cần dependency:

- Số nguyên, số thập phân, phân số, khoảng số, số âm; đúng luật `mốt` / `lăm` / `tư`
  và `một trăm lẻ năm`, `một nghìn không trăm lẻ năm`
- Ngày tháng (`12/3/2024`), giờ (`14:30`, `7h30`), phần trăm, tiền tệ, nhiệt độ, đơn vị đo
- Số thứ tự (`thứ 4` → `thứ tư`), số La Mã theo ngữ cảnh (`Chương IV`)
- Số điện thoại và dãy số dài đọc từng chữ số
- Từ viết tắt, ký hiệu (`&`, `+`, `%`…), đánh vần từ viết hoa lạ
- NFC, bỏ ký tự ngoài bộ chữ Latin, chuẩn hóa dấu câu, luôn có dấu kết câu

Chạy test:

```bash
python3 pipeline/test_vi_norm.py
```

Mở rộng từ điển ngay trong `config.yaml`, không cần sửa code:

```yaml
text:
  abbreviations:
    hcm: "Hồ Chí Minh"
  units:
    dặm: "dặm"
```

Các tùy chọn đáng chú ý: `le_word` (`lẻ` hay `linh`), `thousand_word` (`nghìn` hay `ngàn`),
`four_after_ten` (`tư` hay `bốn`), `decimal_by_digit` (đọc phần thập phân từng chữ số),
`keep_punctuation` (bộ dấu câu giữ lại — phải khớp với symbol set của fork Bert-VITS2 bạn dùng).

## Lọc chất lượng

Lý do loại được ghi vào `dataset/rejected.jsonl` để kiểm tra lại:
`duration_too_short`, `duration_too_long`, `text_too_short`, `speech_rate_low`,
`speech_rate_high`, `clipping`, `low_snr`, `low_asr_confidence`, `low_word_confidence`,
`no_speech`, `empty_text`, `duplicate_text`.

`low_asr_confidence` dùng `asr_prob` là **trung bình** probability các từ, nên một từ sai lẻ loi
giữa một clip tốt sẽ không bị bắt. `min_word_prob` lọc theo từ **tệ nhất** trong clip, mặc định để
`null` (tắt) vì nó loại khá mạnh — đặt thử `0.4` nếu bạn muốn siết.

`speech_rate_low` / `speech_rate_high` là bộ lọc quan trọng nhất: tỉ lệ ký tự trên giây bất thường
thường có nghĩa là text và audio không khớp — đúng loại lỗi làm model học sai.

## Lưu ý

**Nguồn 128 kbps AAC.** File `original/sample.mp3` là AAC 128 kbps — đã mất dữ liệu ở tần số cao,
đây là mức trần chất lượng của model. Nếu có bản gốc WAV/FLAC hoặc bitrate cao hơn thì nên dùng bản đó.

**Bert-VITS2 gốc không có tiếng Việt.** Bản chính thức chỉ hỗ trợ `ZH` / `JP` / `EN` với BERT tương ứng.
Muốn train tiếng Việt cần fork có thêm symbol set + g2p tiếng Việt và một BERT tiếng Việt
(thường là `vinai/phobert-base`). Trường `project.language` trong config đang để `VI` — sửa cho khớp
với fork bạn dùng. `metadata.csv` giúp chuyển sang trainer khác mà không phải chạy lại pipeline.

**Khử nhiễu nhẹ thôi.** `strength: 0.6` là mặc định có chủ ý. Khử nhiễu mạnh tạo artifact và model sẽ
học luôn cả artifact đó. Nếu giọng thu sạch, dùng `--denoise none` thường cho kết quả tốt hơn.
Chỉ dùng `--denoise demucs` khi nền có nhạc.

**Thời lượng.** 57 phút cho ra khoảng 25–35 phút audio dùng được — đủ để fine-tune, chưa đủ để train
từ đầu. Muốn giọng ổn định nên có 3+ giờ cùng một người đọc, cùng thiết bị thu.
