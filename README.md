# PROJEKT Z UCZENIA MASZYNOWEGO W PYTHON 2022

## Wykonali:

- Krzysztof Gońda 25738

- Wojciech Pachciarek 25686

- Marcin Herman 26253

## Opis klas:

### 1. Data:

Główna klasa realizująca całą logikę kodu.
Przechowuje obiekt DataFrame wygenerowany na podstawie pliku csv.

### 2. Parser:

Klasa odpowiada za parsowanie danych testowych na dane liczbowe.

### 3. Train:

Klasa odpowiada za wywołanie algorytmów uczenia maszynowego.
Pozwala na przygotowanie, trenowanie oraz porównanie wytrenowanych modeli.
Wykorzystuje algorytmy support-vector machines oraz Random Forest.
Realizuje również podstawowy find tuting i generuje wiele modeli, a następnie porównuje ich dokładnośc i wybiera
najlepszy z nich.

### 4. CollegeEntry:

Klasa reprezentująca encję danych, wykorzystywana jest do predyktowania za pomocą wytrenowanego modelu.

### 5. Klasy Testujące:

Klasy odpowiadające za uruchamianie testów jednostkowych.

## Źródło danych:

https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset?fbclid=IwAR2SfffmDMFQ4hJNEn9Is4yAD1tJpDnsYDTha2DhCrOXZOdpbdoqfip9ZQY

## Opis bazy danych:

W bazie danych znajdują się sztuczne dane wygenerowane na
potrzeby projektu uczelnianego.
Dane te mają na celu przewidzieć, czy studenci będą kontynuować naukę w college'u czy nie

#### Zgromadzone dane zawierają:

- type_school - Typ szkoły, do której uczęszcza uczeń

    - Academic - Liceum
    - Vocational - szkoła zawodowa

- school_accreditation - Status szkoły. A jest lepsza niż B.

    - A
    - B

- gender - Płeć

    - Male - Mężczyzna
    - Female - Kobieta
- interest - Jak bardzo uczniowie są zainteresowani tym, czy pójdą na studia.
    - Very intrested - Bardzo zainteresowany
    - Quite intresteed - Całkiem zainteresowany
    - less intrested - Mniej zainteresowany
    - Uncertain - Niepewny
    - Not intrested - Niezainteresowany

- residence - Obszar zamieszkania
    - Urban - miejski
    - Rural - wiejski

- parent_age - wiek rodziców (int)
- parent_salary - zarobki rodziców w Rupiach Indonezyjskich
- house_area -
  Powierzchnia domu rodzinnego w metrach kwadratowych
- average_grades - Średnia ocen w skali 0-100
- parent_was_in_college -
  Czy rodzic był kiedykolwiek w college'u?
    - True
    - False

## Wnioski:

Na podstawie wybranej przez nas bazy danych zauważyliśmy że w naszym przypadku najlepiej sprawdza się algorytm Random
Forest z parametrem min_samples_split ustawionym na 2 lub criterion ustawionym na "entropy".

Po przeanalizowaniu wyników oraz tabeli korelacji, mnożna zauważyć że na podstawie zgromadzonych danych,
największy wpływ na to czy obiekt badań pójdzie na studia świadczy zamożność jego rodziców, wielkość mieszkania oraz
średnia ocen.

Przy próbach przewidywania wyniki nie są w pełni dokładne, jednak najczęściej największy wpływ zdają się mieć oceny. 
Często uczeń mający wysokie wyniki w nauce nie potrzebuje innych wysokich wartości, jednak zdarzają się odejścia od normy. 
Duża ilości pieniędzy rodziców, rozmiar mieszkania również potrafią spowodować że uczeń zostanie zklasyfikowany jako
chętnego na pójście na studia. Pozostałe wartości zdają się nie mieć wpływu na ostateczny wynik.
