# Praca magisterska

## Analiza porównawcza wybranych dużych modeli językowych w kontekście wspomagania nauki języka obcego

Repozytorium zawiera kod, prompty oraz narzędzia wykorzystane podczas realizacji pracy magisterskiej poświęconej analizie możliwości zastosowania lokalnie uruchamianych dużych modeli językowych (LLM) do wspomagania nauki języka angielskiego.

## Cel projektu

Celem projektu jest porównanie wybranych dużych modeli językowych pod względem ich przydatności do automatycznego przygotowywania materiałów wspomagających naukę języka obcego.

Analiza obejmuje zarówno techniczną poprawność generowanych odpowiedzi, jak i ich jakość merytoryczną oraz dydaktyczną.

W ramach eksperymentu analizowano:

* **skuteczność generowania odpowiedzi**, w tym odsetek prób zakończonych poprawnym uzyskaniem odpowiedzi od modelu,
* **konieczność przetwarzania odpowiedzi**, w szczególności oczyszczania wygenerowanego tekstu przed próbą odczytu danych w formacie JSON,
* **poprawność strukturalną odpowiedzi**, sprawdzaną za pomocą automatycznej walidacji zgodności z wymaganym schematem,
* **rodzaje i częstość błędów walidacji**, z rozróżnieniem błędów fatalnych oraz odstępstw możliwych do zaakceptowania w trybie walidacji tolerancyjnej,
* **parametry techniczne generowania**, obejmujące m.in. czas wykonania próby, szybkość generowania tokenów oraz informacje zwracane przez środowisko Ollama,
* **parametry związane z wykorzystaniem GPU**, w tym użycie pamięci VRAM, obciążenie karty graficznej, pobór mocy oraz szacowane zużycie energii,
* **jakość odpowiedzi zakwalifikowanych do dalszej oceny**, analizowaną przy użyciu modelu-sędziego pod względem zgodności z poleceniem, poprawności merytorycznej i wartości dydaktycznej,
* **praktyczną użyteczność odpowiedzi**, określaną przez model-sędziego w postaci decyzji, czy dana odpowiedź może zostać bezpośrednio wykorzystana przez osobę uczącą się.

## Badane modele

W eksperymencie wykorzystano pięć dużych modeli językowych:

* **Bielik-4.5B-v3.0-Instruct**
* **DeepSeek-R1 8B**
* **Llama 3.1 8B**
* **Ministral 3 3B**
* **Qwen3.5 4B**

Modele uruchamiano lokalnie przy wykorzystaniu środowiska **Ollama**.


## Scenariusze badawcze

Modele testowano pod kątem generowania różnych rodzajów materiałów dydaktycznych wspomagających naukę języka angielskiego.

### Gramatyka

Generowanie ćwiczeń dotyczących m.in.:

* okresów warunkowych,
* konstrukcji *gerund vs infinitive*,
* *Past Simple vs Present Perfect*,
* czasów prostych i ciągłych.

### Słownictwo

Generowanie ćwiczeń obejmujących m.in.:

* synonimy,
* antonimy,
* dopasowywanie słów do znaczeń,
* wskazywanie poprawnych definicji.

### Wypowiedź pisemna

Analiza oraz korekta przykładowych wypowiedzi pisemnych ucznia wraz z przygotowaniem informacji zwrotnej.

### Test poziomujący

Generowanie pytań umożliwiających ocenę znajomości języka angielskiego na różnych poziomach CEFR.


## Struktura repozytorium

Repozytorium zostało podzielone na katalogi odpowiadające kolejnym etapom eksperymentu:

- **`aggregated/`** – zawiera zagregowane wyniki: eksperymentu i automatycznej walidacji strukturalnej poszczególnych modeli i trybów badawczych.
- **`judge_pipeline/`** – zawiera skrypty i pliki związane z oceną jakościową odpowiedzi przy użyciu modelu-sędziego.
- **`logs/`** – zawiera logi z eksperymentu i automatycznej walidacji strukturalnej.
- **`prompts/`** – zawiera prompty systemowe i użytkownika wykorzystywane w poszczególnych scenariuszach badawczych.
- **`results/`** – zawiera wyniki poszczególnych prób generowania odpowiedzi przez badane modele wraz z zapisanymi metrykami i wynikami automatycznej walidacji strukturalnej.
- **`validation/`** – zawiera skrypty i pliki związane z automatyczną walidacją strukturalną odpowiedzi.

## Dodatkowe informacje

- Eksperymenty podstawowe przeprowadzono lokalnie z wykorzystaniem osobistej karty graficznej **NVIDIA GeForce RTX 2060 Max-Q 6 GB VRAM**.

- Do części związanej z oceną jakościową odpowiedzi wykorzystano środowisko wyposażone w kartę **NVIDIA A100 80 GB PCIe**.
  
- Do oceny jakościowej wykorzystano dodatkowy model językowy Qwen/Qwen3.6-27B pełniący funkcję modelu-sędziego.

## Charakter projektu

Repozytorium stanowi część pracy magisterskiej i zostało przygotowane przede wszystkim w celu zapewnienia powtarzalności przeprowadzonych eksperymentów oraz udokumentowania zastosowanej metodyki badawczej.

---

**Autor:** Anna Michalak
**Rodzaj projektu:** praca magisterska
**Tematyka:** duże modele językowe, przetwarzanie języka naturalnego, wspomaganie nauki języków, inżynieria promptów

