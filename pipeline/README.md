# Pipeline tạo dataset TTS tiếng Việt

Chạy mọi lệnh từ thư mục gốc dự án. Giải thích cách hoạt động: [NOTES.md](NOTES.md).

## Cài đặt

```bash
bash pipeline/setup.sh
```

## Chạy

Thử 5 phút đầu của mỗi file trước:

```bash
pipeline/.venv/bin/python pipeline/run.py --preview-minutes 5
```

Chạy toàn bộ:

```bash
pipeline/.venv/bin/python pipeline/run.py --force
```

Hai dòng log cần kiểm tra: `align: sau khi can chinh, N% ... (truoc M%)` phải có N lớn hơn M, và
`segment: kiem tra OK`.

## Thêm file audio mới

Thả file vào `original/` rồi chạy lại lệnh trên. File mới được xử lý, file cũ bỏ qua, dataset dựng
lại từ tất cả. Tất cả file phải **cùng một người đọc**.

Chạy riêng một file:

```bash
pipeline/.venv/bin/python pipeline/run.py --only sample03
pipeline/.venv/bin/python pipeline/run.py --only sample02 --force --stages decode,vad,denoise,asr,align,segment,cut,text,qc
...
pipeline/.venv/bin/python pipeline/run.py
```

## Sửa text bằng tay

Mở `work/<mã nguồn>/edit.txt`, sửa dòng text dưới mỗi dòng `#`:

```
# sample_00009.wav  6.60s  ctc=0.84  daydu=1.02
chiến đấu với quang Thái Bình Tuyên Quốc.
```

Rồi chạy:

```bash
pipeline/.venv/bin/python pipeline/run.py --stages qc,export --force
```

- Tự viết số thành chữ: `mười lăm tuổi`, không phải `15 tuổi`
- Dấu câu chỉ dùng `, . ! ? …`
- Đừng sửa `dataset/Data/<name>/filelists/*.list`, nó bị ghi lại mỗi lần export
- `edit.txt` không bao giờ bị pipeline ghi đè. Xóa file để bỏ hết sửa tay

## Nghe thử một clip

```bash
afplay dataset/Data/vi_story/wavs/sample_00009.wav
afplay work/sample/clips/sample_00000.wav
```

## Tham số

| Cờ | Ý nghĩa |
| --- | --- |
| `--preview-minutes N` | Chỉ xử lý N phút đầu mỗi file |
| `--force` | Chạy lại cả bước đã có kết quả |
| `--only a,b` | Chỉ xử lý các nguồn này |
| `--stages a,b` | `decode,vad,denoise,asr,align,segment,cut,text,qc,export` |
| `--denoise X` | `none` / `ffmpeg` / `noisereduce` / `demucs` |
| `--asr X` | `mlx` / `faster-whisper` |
| `--speaker NAME` | Đổi tên speaker |

## Đầu ra

```
dataset/
├── Data/vi_story/
│   ├── wavs/*.wav                  bản 44.1 kHz dùng luôn được
│   ├── raw/narrator/*.wav          nguồn cho resample.py của Bert-VITS2
│   └── filelists/narrator.list     path|speaker|VI|text
├── metadata.csv                    LJSpeech, cho GPT-SoVITS / XTTS
├── report.md                       thống kê từng file nguồn
└── rejected.jsonl                  clip bị loại kèm lý do
```

`work/<mã nguồn>/` giữ file trung gian để chạy lại từng bước. Xóa được, nhưng xóa rồi thì phải
transcribe lại từ đầu.

## Learning

```
afplay -r 0.85 work/sample/clips/sample_00584.wav
python f5tts/prepare_dataset.py
python3 f5tts/monitor_server.py
bash f5tts/run_paced.sh

ps aux | grep -E "finetune_cli|run_paced|accelerate" | grep -v grep
kill -9 <PID>

nohup caffeinate -i -w 38722 > /dev/null 2>&1 &
pmset -g assertions | grep caffeinate

cd f5tts/F5-TTS-Vietnamese/data && zip -r ~/your_training_dataset.zip your_training_dataset
```