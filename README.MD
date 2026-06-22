# Spis treści
- [Spis treści](#spis-treści)
- [Pobranie repozytorium oraz stworzenie i uruchomienie środowiska wykonawczego](#pobranie-repozytorium-oraz-stworzenie-i-uruchomienie-środowiska-wykonawczego)
- [Rozpoznawanie pozycji szachowej z obrazu](#rozpoznawanie-pozycji-szachowej-z-obrazu)
- [Opis projektu](#opis-projektu)
- [Architektura systemu](#architektura-systemu)
  - [1. Model wykrywania pól](#1-model-wykrywania-pól)
  - [2. Model rozpoznawania figur](#2-model-rozpoznawania-figur)
  - [3. Rekonstrukcja pozycji](#3-rekonstrukcja-pozycji)
- [Dataset](#dataset)
- [Struktura repozytorium](#struktura-repozytorium)
- [Licencja](#licencja)
- [Wyniki](#wyniki)


# Pobranie repozytorium oraz stworzenie i uruchomienie środowiska wykonawczego

Pobranie repozytorium.
```bash
git clone https://github.com/KrzysztofEmerling/ChessPiecesCV.git
cd ChessPiecesCV
```

Stworzenie i uruchomienie środowiska wykonawczego (Linux/macOS).
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

# Rozpoznawanie pozycji szachowej z obrazu

Projekt ma na celu rozpoznawanie pozycji figur szachowych na podstawie obrazu lub nagrania wideo z wykorzystaniem metod analizy obrazu.

System wykrywa szachownicę, dzieli ją na pojedyncze pola, a następnie określa, jaka figura znajduje się na każdym polu. Na tej podstawie możliwe jest odtworzenie pełnej pozycji szachowej.

---

# Opis projektu

Projekt składa się z trzech głównych etapów:

1. **Wykrywanie szachownicy i pól**
2. **Rozpoznawanie figur**
3. **Rekonstrukcja pozycji na szachownicy**

Połączenie tych elementów pozwala określić pełną pozycję szachową na podstawie pojedynczego zdjęcia lub klatki wideo.

---

# Architektura systemu

## 1. Model wykrywania pól

Pierwszy model odpowiada za wykrycie szachownicy na obrazie oraz podział jej na 64 pola.

Zadania modelu:

- wykrycie szachownicy na obrazie
- korekcja perspektywy
- podział planszy na 64 pola
- zapis obrazów pojedynczych pól

Wygenerowane obrazy pól są następnie wykorzystywane do stworzenia datasetu do rozpoznawania figur.

---

## 2. Model rozpoznawania figur

Na podstawie wygenerowanego datasetu trenowany jest drugi model, którego zadaniem jest określenie zawartości danego pola.

Zamiast ciężkich modeli deep learning wykorzystywane są **embeddingi cech obrazu**, które następnie trafiają do klasyfikatora.

Schemat działania:

obraz pola → wektor cech (embedding) → klasyfikator → typ figury

Przykładowe klasy:

- puste pole
- białe figury
- czarne figury

---

## 3. Rekonstrukcja pozycji

Ostatni etap łączy wyniki obu modeli.

Pipeline systemu:

obraz / klatka wideo
→ wykrycie szachownicy
→ podział na pola
→ klasyfikacja figur
→ rekonstrukcja pozycji na planszy

System docelowo może działać na pojedynczych zdjęciach

---

# Dataset

Projekt wykorzystuje zewnętrzne zbiory danych z obrazami szachownic.

Źródło datasetu: <NAZWA DATASETU>

Autor / organizacja: <AUTOR>

Licencja:
Creative Commons Attribution-NonCommercial-ShareAlike 4.0

https://creativecommons.org/licenses/by-nc-sa/4.0/

Dataset **nie jest dystrybuowany w tym repozytorium**.
Należy go pobrać bezpośrednio z oryginalnego źródła.

Dane są wykorzystywane do trenowania modeli wykrywania pól oraz klasyfikacji figur.

---

# Struktura repozytorium

```
project/
│
├── annotation/
│   ręcznie wykonane adnotacje plansz
│
├── board_detection/
│   model i kod do wykrywania szachownicy oraz pól
│
├── piece_recognition/
│   generowanie embeddingów i klasyfikacja figur
│
├── dataset_generation/
│   skrypty generujące dataset pól z obrazów planszy
│
├── inference/
│   końcowy pipeline łączący oba modele
│
├── manual_annotations/
│   narzędzie do ręcznego anotowania plansz
│
└── experiments/
    eksperymenty i notatniki treningowe
```

---

# Licencja

Repozytorium zawiera wyłącznie kod.

Datasety wykorzystywane w projekcie podlegają własnym licencjom.
Szczegóły znajdują się w sekcji **Dataset**.

---

# Wyniki

- Kod prawidłowo realizuje korekcję perspektywy wraz z wykrywaniem szachownicy, w przypadku braku wykrycia szachwonicy program zwróci błąd.
- W repozytorium porównano 3 różne modele wykrywania typów figur szachowych LSB + RandomForest, MPL, CNN, jednakże z uwagi na niezbalansowanie nie udało uzyskać się wystarczającej dokładności w żadnym z zastosowanych modeli. Na niepowodzenie części ML wpłynęły brak uwzględnienia rotacji planszy powyżej 30 deg (odwracającej szachownicę). Wpłynęło to na niepoprawne zaetykietowanie części danych treningowych. Skutkiem czego dokładność modeli wynosi około 70% gdzie modele MPL i CNN działały nieznacznie lepiej niż prostszy korzystający z RandomForest. Niska dokładność modeli nie pozwoliła na prawidłową rekonstrukcji pozycji szachowej.
- W repozytorium dokonano porównania obrazów oznaczonych ręcznie z modelami oznaczonymi przez utworzoną funkcję, uzyskując 10% pokrycia.
- na końcu połączono metodę korekcji perspektywy wraz z wykrywaniem pól szachownicy z modelem MPL, zapisującą szachownicę do formatu FEN. Prezentującą wyniki połączenia wszystkich elementów projektu.
