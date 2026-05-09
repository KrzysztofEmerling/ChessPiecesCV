# Instrukcja obsługi `annotate_chessboard.py`

## Ogólny opis

Program `annotate_chessboard.py` służy do ręcznego anotowania narożników szachownicy w zbiorze obrazów. Program:

- automatycznie permutuje obrazy,
- wykrywa kandydujące punkty przecięć szachownicy za pomocą OpenCV,
- wyświetla interfejs graficzny do ręcznego klikania 4 narożników planszy,
- automatycznie „przyciąga" klikięcia do bliskich wykrytych punktów,
- zapisuje anotacje w formacie JSON w folderze `Dataset/Validation`.

## Wymagania

- Python 3.11+
- biblioteki: `cv2`, `numpy`
- źródłowe obrazy szachownic w jednym folderze

## Uruchamianie

### Podstawowe użycie

```powershell
python annotate_chessboard.py ścieżka/do/obrazów
```

### Pełny przykład z opcjami

```powershell
python annotate_chessboard.py dataset/chessred/images `
  --output Dataset/Validation `
  --seed 42 `
  --limit 10 `
  --copy-images
```

## Dostępne opcje

| Opcja | Wartość domyślna | Opis |
|-------|------------------|------|
| `--output` | `Dataset/Validation` | Folder wyjściowy na pliki JSON |
| `--seed` | `42` | Ziarno do permutacji (dla powtarzalności) |
| `--limit` | brak | Maksymalna liczba obrazów do anotacji |
| `--debug` | wyłączony | Wyświetla okno debug'u z wykrytymi punktami |
| `--copy-images` | wyłączony | Kopiuje oryginalne obrazy do folderu wyjściowego |

## Obsługa interfejsu

Podczas anotacji obrazu zobaczysz okno z:
- **żółtymi kółkami** – kandydujące punkty przecięć (wykryte automatycznie),
- **zielonymi kółkami** – wybrane narożniki (aktualne).

### Klawisze

| Klawisz | Akcja |
|---------|-------|
| **LPM** (lewy przycisk myszy) | Kliknij na narożnik planszy |
| **s** | Zapisz anotację (wymaga 4 wybranych narożników) |
| **r** | Resetuj wybór (wyczyść wszystkie narożniki) |
| **d** | Usuń ostatnio wybrany narożnik |
| **n** | Pomiń ten obraz |
| **q** | Wyjdź z programu |

## Workflow

1. **Klikanie narożników**: Program wyświetla obraz i prosi o kliknięcie 4 narożników planszy w kolejności:
   - top-left (górny-lewy)
   - top-right (górny-prawy)
   - bottom-right (dolny-prawy)
   - bottom-left (dolny-lewy)

2. **Auto-snap**: Jeśli klikniesz blisko wykrytego punktu, program automatycznie "przyciąga" Twoje kliknięcie do tego punktu.

3. **Zapis**: Po wyborze 4 narożników naciśnij `s`, aby zapisać anotację.

4. **Format zapisu**: Dla obrazu `obraz.jpg` zostanie zapisany plik `Dataset/Validation/obraz.json` zawierający:
   ```json
   {
     "source_image": "ścieżka/do/obraz.jpg",
     "corners": [
       {"x": 123, "y": 45},
       {"x": 456, "y": 78},
       {"x": 456, "y": 210},
       {"x": 123, "y": 210}
     ],
     "candidates": [
       {"x": 120, "y": 42},
       ...
     ]
   }
   ```

## Przykłady

### Anotacja pierwszych 5 obrazów z debug'iem

```powershell
python annotate_chessboard.py dataset/images --limit 5 --debug
```

### Anotacja wszystkich obrazów i kopiowanie do folderu walidacji

```powershell
python annotate_chessboard.py dataset/images --copy-images
```

### Używanie innego ziarna (dla innej kolejności)

```powershell
python annotate_chessboard.py dataset/images --seed 123
```

## Wskazówki

- **Jeśli program nie widzi przecięć**: Sprawdź, czy obraz zawiera wyraźne linie szachownicy. Może pomóc opcja `--debug`.
- **Jeśli chcesz pominąć obraz**: Naciśnij `n`.
- **Jeśli pomyliłeś narożnik**: Naciśnij `d`, aby usunąć ostatnią kliknięcie.
- **Jeśli chcesz resetować**: Naciśnij `r`, aby zacząć od nowa.
- **Jeśli program się zawiesza**: Naciśnij `q` (quit).

## Struktura folderu wyjściowego

Po anotacji struktura będzie wyglądać tak:

```
Dataset/Validation/
├── obraz_1.json
├── obraz_2.json
├── obraz_1.jpg         (jeśli użyto --copy-images)
├── obraz_2.jpg         (jeśli użyto --copy-images)
└── ...
```

## Troubleshooting

**Problem**: `No image files found`
- **Rozwiązanie**: Sprawdź, czy podana ścieżka istnieje i zawiera obrazy (.jpg, .png, itd.)

**Problem**: Program zamyka się natychmiast
- **Rozwiązanie**: Sprawdź konsoli błąd – może brakować biblioteki (np. `cv2`). Zainstaluj: `pip install opencv-python`

**Problem**: Interfejs nie reaguje na kliknięcia
- **Rozwiązanie**: Kliknij w okno, aby upewnić się, że ma fokus, a następnie kliknij ponownie.

---

**Autor**: Projekt ChessPiecesCV  
**Data**: Maj 2026
