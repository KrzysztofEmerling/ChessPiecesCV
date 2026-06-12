tutorial do generowania -Twojej starej- odseparowanych pól szachownicy

Bez etykiet (wszystko do `_unlabeled/`):

```
python3 -m dataset_generation.generate_squares \
    --image-root dataset/chessred2k/images \
    --output dataset_squares
```

Z etykietami ChessReD:

```
python3 -m dataset_generation.generate_squares \
    --image-root dataset/chessred2k/images \
    --chessred-annotations dataset/chessred2k/annotations.json \
    --output dataset_squares
```

Szybki test na 10 obrazach:

```
python3 -m dataset_generation.generate_squares --limit 10 --output /tmp/squares
```

## Flagi

Chat generował tę część readme, ale wygląda G, więc można używać, wszystkie te flagi działają.

| Flaga | Opis |
|-------|------|
| `--corners-dir DIR` | Folder z adnotacjami narożników (domyślnie `annotations/`) |
| `--image-root DIR` | Folder z obrazami źródłowymi (domyślnie: ścieżka z pola `source_image`) |
| `--output DIR` | Folder wyjściowy (domyślnie `dataset_squares/`) |
| `--chessred-annotations FILE` | `annotations.json` z ChessReD → etykiety figur |
| `--padding N` | Obetnij N px z każdej krawędzi pola (eliminuje linie siatki) |
| `--flip` | Obrót planszy 180° (jeśli annotator oznaczył a1 jako TL zamiast a8) |
| `--limit N` | Przetwórz najwyżej N obrazów |
| `--fallback-detect` | Gdy brak adnotacji — użyj `detect_board()` z `board_detection/` |