# Changes made by Embryo-Zero (Embryo Zero Backyard Data Center Corporation) to faster-whisper

Forked from v1.2.1-19-g6cb639e (commit 6cb639e) on 2026-11-16. All changes are on the `embryo` branch; `git log` has the full detail.

| Date | Files | Change |
|---|---|---|
| 2026-11-16 | EMBRYO.md, LEGAL.md, CHANGES-EMBRYO.md (new) | Fork notes, legal notes and this list. No original file changed yet. |
| 2026-11-16 | README.md | Embryo name at the top; the original README follows unchanged |

## 2026-11-24
- embryo/embryo_captions.py: the embryo-captions command (audio or video in, .srt + .vtt + timed JSON out). Runs on the
  processor (int8) by default; EMBRYO_CAPTIONS_GPU=1 tries the card. Used by Embryo Movie Maker to caption speech in
  uploaded clips whose sound is kept. Model: Whisper small (MIT), not stored in this repository.
