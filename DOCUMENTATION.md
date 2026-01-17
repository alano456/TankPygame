# Dokumentacja Projektu TankPygame

## Przegląd Struktury Projektu
Projekt jest implementacją środowiska gry "Czołgi" (Tanks) zintegrowanego z biblioteką `gymnasium`, umożliwiającego trening agentów Reinforcement Learning (RL) takich jak SARSA, Q-Learning i DQN.

## Główna Struktura Katalogów
- **root/**: Główne skrypty uruchomieniowe i treningowe.
- **env/**: Definicja środowiska Gym.
- **agents/**: Implementacje agentów RL.
- **game/**: Logika gry (fizyka, obiekty).
- **logs_*/:** Katalogi z logami treningowymi i zapisanymi modelami.

---

## 1. Główne Skrypty i Pliki Wejściowe

### `main.py`
**Rola**: Główny plik uruchomieniowy gry. Pozwala na grę człowiek vs bot lub obserwację walki AI vs AI.
**Kluczowe Funkcje**:
- `start_game()`: Inicjalizuje środowisko gry.
- `run_game()`: Główna pętla gry (renderowanie, obsługa wejścia).
- `load_agent()`: Wczytuje wytrenowane modele (QLearning, SARSA, DQN) z plików `.pkl` lub `.pth`.
- **Integracja**: Przekazuje stan gry do agentów, odbiera akcje, aktualizuje środowisko i renderuje obraz przy użyciu Pygame.
- **Sensorium**: Dla gracza "Człowiek" mapuje klawisze strzałek na akcje. Dla agentów przekazuje stan (obserwację) i wykonuje predykcję.

### `train_sarsa.py`
**Rola**: Skrypt dedykowany do treningu agenta SARSA z wykorzystaniem Curriculum Learning.
**Kluczowe Elementy**:
- `TrainingTankEnv`: Klasa dziedzicząca po `TankEnv`, nadpisująca logikę `reset` i `step` dla potrzeb treningu (np. fazy trudności).
- **Curriculum Learning**: Implementuje 3 fazy treningu (Tarcza, Ruchomy Cel, Killer) w zależności od numeru epizodu.
- **Reward Shaping**: Definiuje złożony system nagród (za celowanie, zbliżanie się, trafienie, karę za strzał w ścianę).
- **Pętla Treningowa**: Wykonuje algorytm SARSA (Update Q-Table: $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$).
- **Logowanie**: Zapisuje postępy do `logs_sarsa/training_log.csv`.

### `analyze_training.py`
**Rola**: Narzędzie analityczne do wizualizacji postępów treningu.
**Funkcje**:
- Generuje wykresy nagród, win-rate i długości epizodów.
- Tworzy heatmapy (mapy ciepła) pozycji i śmierci agenta.
- Generuje raport podsumowujący w pliku Markdown.

---

## 2. Środowisko (`env/`)

### `env/tank_env.py`
**Rola**: Serce symulacji. Implementuje interfejs `gym.Env`.
**Kluczowe Klasy**:
- `TankEnv`: Główna klasa środowiska.
    - `reset()`: Restartuje grę, losuje pozycje.
    - `step(action)`: Wykonuje krok symulacji. Oblicza fizykę pocisków, kolizje i zwraca `(observation, reward, terminated, truncated, info)`.
    - `_get_obs()`: Zwraca wektor stanu (pozycje, HP, sensory).
    - `action_space`: Dyskretna przestrzeń akcji (0: Lewo, 1: Prawo, 2: Przód, 3: Strzał).
    - `observation_space`: 14-elementowy wektor (pozycje gracza i wroga, kierunki, sensorow w ścianach).

---

## 3. Agenci (`agents/`)

### `agents/sarsa_agent.py`
**Rola**: Implementacja algorytmu SARSA.
**Kluczowe Funkcje**:
- `get_state_key(state)`: Dyskretyzuje ciągły/złożony stan gry na zestaw cech użytecznych dla Q-Table (np. Kąt do wroga, Dystans, Czy widzę wroga?).
    - **Cechy**: `angle_bin` (kąt), `dist_bin` (odległość), `can_shoot` (cooldown), `is_threat` (czy wróg celuje), `has_los` (widoczność).
- `get_action(state, epsilon)`: Wybiera akcję zgodnie ze strategią Epsilon-Greedy.
- `update(...)`: Aktualizuje wartości w tabeli Q na podstawie równania SARSA.

---

## 4. Logika Gry (`game/`)

### `game/tank.py`
**Rola**: Definicja obiektu czołgu.
**Zmienne**: `x`, `y` (pozycja), `direction` (kierunek N/E/S/W), `hp` (punkty życia).
**Funkcje**: `move_forward()`, `turn_left()`, `turn_right()`.

### `game/grid.py`
**Rola**: Zarządzanie mapą i kolizjami.
**Funkcje**:
- `is_free(x, y)`: Sprawdza czy pole jest puste.
- `clear_line(x1, y1, x2, y2)`: Sprawdza "Line of Sight" (czy linia między punktami nie przecina ściany). Kluczowe dla sensora widoczności agenta.

### `game/enemy.py`
**Rola**: Prosta sztuczna inteligencja (Rule-Based) dla przeciwnika (EnemeyBot).
**Funkcje**:
- `choose_action()`: Decyduje o ruchu bota (podejdź, obróć się, strzel). Używana jako "sparingpartner" podczas treningu.

---

## Przepływ Treningu (Workflow)
1. **Konfiguracja**: Ustawienie hiperparametrów w `train_sarsa.py` (ilość epizodów, fazy curriculum).
2. **Uruchomienie**: `python train_sarsa.py`.
3. **Symulacja**: Skrypt uruchamia `TrainingTankEnv`, agent podejmuje decyzje, środowisko zwraca nagrody.
4. **Zapis**: Po zakończeniu, model zapisywany jest jako `sarsa_agent.pkl`.
5. **Weryfikacja**: Uruchomienie `main.py` (Agent Type: SARSA) by zobaczyć efekt.
6. **Analiza**: `python analyze_training.py logs_sarsa` generuje wykresy.
