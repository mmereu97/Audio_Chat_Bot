# 🤝 Ghid de Contribuție

Mulțumim pentru interesul de a contribui la **Chat Vocal Avansat cu Gemini AI**! 🎉

Acest document oferă ghiduri și best practices pentru a face contribuțiile tale cat mai ușoare și eficiente.

---

## 📋 Cuprins

- [Cod de Conduită](#-cod-de-conduită)
- [Cum pot contribui?](#-cum-pot-contribui)
- [Setup pentru Dezvoltare](#-setup-pentru-dezvoltare)
- [Procesul de Contribuție](#-procesul-de-contribuție)
- [Stilul de Cod](#-stilul-de-cod)
- [Commit Messages](#-commit-messages)
- [Pull Request Process](#-pull-request-process)
- [Raportare Bug-uri](#-raportare-bug-uri)
- [Cereri de Feature-uri](#-cereri-de-feature-uri)

---

## 📜 Cod de Conduită

Participând la acest proiect, te angajezi să menții un mediu prietenos, respectuos și incluziv pentru toată lumea.

### Așteptări:
- ✅ Folosește un limbaj primitor și incluziv
- ✅ Respectă punctele de vedere și experiențele diferite
- ✅ Acceptă cu grație critica constructivă
- ✅ Concentrează-te pe ce e cel mai bine pentru comunitate
- ✅ Arată empatie față de alți membri ai comunității

### Nu sunt acceptate:
- ❌ Limbaj sau imagini sexualizate
- ❌ Trolling, insulte sau comentarii depreciative
- ❌ Hărțuire publică sau privată
- ❌ Publicarea informațiilor private ale altora
- ❌ Alte comportamente care ar putea fi considerate nepotrivite

---

## 🎯 Cum pot contribui?

Există multe moduri de a contribui la proiect:

### 1. 🐛 Raportează Bug-uri
Găsit un bug? Deschide un **Issue** pe GitHub cu:
- Descriere clară a problemei
- Pași pentru reproducere
- Comportament așteptat vs. comportament actual
- Screenshots/logs dacă e relevant
- Informații despre sistem (OS, Python version, etc.)

### 2. 💡 Propune Feature-uri Noi
Ai o idee grozavă? Deschide un **Issue** cu:
- Descriere detaliată a feature-ului
- De ce ar fi util?
- Exemple de use cases
- Mockup-uri sau diagrame (opțional)

### 3. 📝 Îmbunătățește Documentația
- Corectează erori de scriere sau gramatică
- Adaugă exemple suplimentare
- Îmbunătățește claritatea explicațiilor
- Traduce documentația în alte limbi

### 4. 💻 Contribuie cu Cod
- Fix bug-uri existente
- Implementează feature-uri noi
- Îmbunătățește performanța
- Adaugă teste
- Refactorizează cod

### 5. 🎨 Design & UX
- Propune îmbunătățiri UI
- Creează icoane sau assets
- Îmbunătățește experiența utilizatorului

---

## 🛠️ Setup pentru Dezvoltare

### Cerințe Preliminare
- Python 3.8 sau mai nou
- Git
- Virtual environment tool (venv sau conda)

### Pas cu pas:

1. **Fork repository-ul**
   ```bash
   # Click pe "Fork" în GitHub UI
   ```

2. **Clone fork-ul tău**
   ```bash
   git clone https://github.com/YOUR-USERNAME/voice-chat-gemini.git
   cd voice-chat-gemini
   ```

3. **Adaugă upstream remote**
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/voice-chat-gemini.git
   ```

4. **Creează virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Instalează dependențele**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Dependențe pentru development (dacă există)
   ```

6. **Configurează `.env`**
   ```bash
   cp .env.example .env
   # Editează .env și adaugă GOOGLE_API_KEY
   ```

7. **Verifică că rulează**
   ```bash
   python voice_chat.py
   ```

---

## 🔄 Procesul de Contribuție

### Workflow Standard:

1. **Sincronizează cu upstream**
   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   ```

2. **Creează un branch nou**
   ```bash
   git checkout -b feature/my-awesome-feature
   # SAU
   git checkout -b bugfix/fix-that-annoying-bug
   ```

3. **Fă modificările**
   - Scrie cod
   - Adaugă teste (dacă e aplicabil)
   - Testează local
   - Actualizează documentația

4. **Commit modificările**
   ```bash
   git add .
   git commit -m "Add: implementare feature X"
   ```

5. **Push pe fork-ul tău**
   ```bash
   git push origin feature/my-awesome-feature
   ```

6. **Deschide Pull Request**
   - Mergi pe GitHub la fork-ul tău
   - Click pe "New Pull Request"
   - Completează template-ul PR

---

## 🎨 Stilul de Cod

### Python Style Guide

Urmărim **PEP 8** cu câteva adaptări:

```python
# ✅ GOOD
def process_audio_data(audio_frames, sample_rate=16000):
    """
    Procesează frame-uri audio și returnează rezultatul.
    
    Args:
        audio_frames (list): Lista de frame-uri audio
        sample_rate (int): Rate-ul de sampling (default: 16000)
    
    Returns:
        np.ndarray: Audio procesat
    """
    processed = np.array(audio_frames)
    return processed

# ❌ BAD
def processAudioData(audioFrames,sampleRate=16000):
    processed=np.array(audioFrames)
    return processed
```

### Naming Conventions

```python
# Classes: PascalCase
class VoiceWorker:
    pass

# Functions/Methods: snake_case
def load_config():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_SPEECH_DURATION = 30

# Private: _leading_underscore
def _internal_helper():
    pass
```

### Docstrings

Folosim docstrings în format Google/NumPy:

```python
def my_function(param1, param2):
    """
    Scurtă descriere a funcției.
    
    Descriere mai detaliată dacă e nevoie.
    Poate avea mai multe paragrafe.
    
    Args:
        param1 (str): Descriere parametru 1
        param2 (int): Descriere parametru 2
    
    Returns:
        bool: True dacă succes, False altfel
    
    Raises:
        ValueError: Dacă param2 este negativ
    """
    pass
```

### Comentarii

```python
# ✅ GOOD - Explică "de ce", nu "ce"
# Folosim threading pentru a evita blocarea UI-ului în timpul procesării
threading.Thread(target=process_data, daemon=True).start()

# ❌ BAD - Evident din cod
# Creăm un thread
threading.Thread(target=process_data, daemon=True).start()
```

### Type Hints (Recomandat)

```python
from typing import List, Optional, Tuple

def get_audio_devices() -> List[str]:
    """Returnează lista de dispozitive audio."""
    pass

def process_text(text: str, max_length: Optional[int] = None) -> Tuple[str, int]:
    """Procesează text și returnează rezultatul."""
    pass
```

---

## 📝 Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: Feature nou
- **fix**: Bug fix
- **docs**: Doar modificări documentație
- **style**: Format, missing semi colons, etc; fără modificări cod
- **refactor**: Refactorizare cod
- **perf**: Îmbunătățiri performanță
- **test**: Adăugare sau modificare teste
- **chore**: Mentenanță (build, dependencies, etc)

### Exemple

```bash
# Feature simplu
feat(audio): add auto-calibration for noise threshold

# Bug fix cu descriere
fix(tts): resolve audio playback stuttering on Windows

Problemă: TTS playback avea hickups pe Windows 11
Soluție: Increased pygame buffer size to 4096

Closes #123

# Refactorizare
refactor(ui): extract semafor logic into separate class

- Created SemaforWidget class
- Moved all semafor-related code from main window
- Added unit tests for new class
```

### Best Practices

- ✅ Folosește timpul prezent: "add" nu "added"
- ✅ Nu capitaliza prima literă
- ✅ Nu pune punct la sfârșit
- ✅ Limită subject la 50 caractere
- ✅ Wrappa body la 72 caractere
- ✅ Explică "ce" și "de ce", nu "cum"

---

## 🔀 Pull Request Process

### Template PR

Când deschizi un PR, completează template-ul:

```markdown
## Description
Descriere clară despre ce face PR-ul.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Descrie testele pe care le-ai făcut:
- [ ] Testat local pe Windows 11
- [ ] Testat cu Python 3.10
- [ ] Testat cu/fără TTS activat
- [ ] etc.

## Checklist
- [ ] Codul urmează style guidelines
- [ ] Am făcut self-review
- [ ] Am comentat părțile complexe
- [ ] Am actualizat documentația
- [ ] Modificările nu generează warning-uri
- [ ] Am adăugat teste (dacă e aplicabil)

## Screenshots (dacă e relevant)
Adaugă screenshots pentru modificări UI.

## Related Issues
Closes #123
References #456
```

### Review Process

1. **Automated Checks** (dacă există CI/CD)
   - Linting
   - Tests
   - Code coverage

2. **Code Review**
   - Cel puțin un maintainer va review
   - Răspunde la feedback constructiv
   - Fă modificările necesare

3. **Merge**
   - După aprobare, un maintainer va face merge
   - Branch-ul va fi șters automat

---

## 🐛 Raportare Bug-uri

### Template Issue pentru Bug

```markdown
**Descriere Bug**
Descriere clară și concisă a bug-ului.

**Pași pentru Reproducere**
1. Pornește aplicația
2. Click pe '...'
3. Scroll down to '...'
4. Vezi eroarea

**Comportament Așteptat**
Ce ar trebui să se întâmple normal.

**Screenshots**
Dacă e aplicabil, adaugă screenshots.

**Environment:**
 - OS: [ex: Windows 11]
 - Python Version: [ex: 3.10.5]
 - App Version: [ex: v1.0.0]

**Logs**
```
Paste relevant logs here
```

**Context Adițional**
Orice altă informație relevantă.
```

### Severitate

Clasifică bug-ul:
- 🔴 **Critical**: App crash, data loss
- 🟠 **High**: Feature major broken
- 🟡 **Medium**: Feature parțial broken
- 🟢 **Low**: Cosmetic, minor issues

---

## 💡 Cereri de Feature-uri

### Template Issue pentru Feature

```markdown
**Descriere Feature**
Descriere clară a feature-ului propus.

**Motivație**
De ce ar fi util acest feature?
Ce problemă rezolvă?

**Soluție Propusă**
Cum vezi implementarea?

**Alternative**
Ai considerat alte abordări?

**Context Adițional**
Screenshots, mockups, exemple din alte apps, etc.
```

---

## 🏆 Recognition

Contributorii vor fi adăugați în:
- README.md - Contributors section
- CHANGELOG.md - pentru contribuții majore
- Release notes

---

## 📞 Întrebări?

Ai întrebări despre contribuție?
- 📧 Trimite email la: your.email@example.com
- 💬 Deschide o **Discussion** pe GitHub
- 🐛 Deschide un **Issue** cu label "question"

---

## 🙏 Mulțumiri

Îți mulțumim pentru contribuție! Fiecare PR, issue, sau sugestie face proiectul mai bun! ❤️

**Happy Coding! 🚀**
