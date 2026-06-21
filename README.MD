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

System docelowo może działać na:

- pojedynczych zdjęciach
- strumieniu z kamery
- nagraniach wideo

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

- Kod prawidłowo realizuje wykrywanie szachownicy, w przypadku brku wykrycia szachwonicy program zwróci błąd.
- W repozytorium porównano 3 różne modele wykrywania typów figór szachowych LSB + RandomForest, MPL, CNN, jednakże z uwagi na niezbalansowanie nie udało uzyskać się wystarczającej dokładności w żadnym z zastosowanych modeli. Na niepowodzenie części ML wpłyneły brak uwzględnienia rotacji planszy powyżej 30 deg (odwracsającej szachownicę). Wpłyneło to na niepoprawne zaetykietowanie części danych treningowych. Skutkiem czego dokładność modeli wynosi około 70% gdzie modele MPL i CNN działały nieznacznie lepiej niż prostrzy kożystający z RandomForest. Niaska dokładność modeli nie pozwoliła na prawidłową rekonstrukcji pozycji szachowej.
- W repozytorium dokonano porównania obrazów oznaczonych ręcznie z modelami oznaczonymi przez utworzoną funkcję, uzyskując 10% pokrycia.
- na końcu połączono metodę korekcji perspektywy wraz z wykrywaniem pól szachwonicy z modelem MPL, zapisującą szachwonicę do formatu FEN. Prezentującą wyniki połączenia wszystkich elementów projektu.
