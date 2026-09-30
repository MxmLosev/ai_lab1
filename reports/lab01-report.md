## 1. Исходное состояние и запуск

### Фактические сведения
- Репозиторий: https://github.com/tylerrussin/Fish-Dimensions-Regression-Analysis.git
- Исходный коммит: 1f65547b63c3043e2dcfff0c6435afa993331d4e (дата: 2022-02-25 06:43:52 -0800)
- Рабочая ветка: lab/01-eda
- Python версия: 3.12.0

### Требования проекта
- Менеджер зависимостей: pip
- Требуемая версия Python (pip): 3.9

### Установка зависимостей
- Подготовка: Сборка зависимостей проекта из pipfile.lock в requirements.txt. Инициализация виртуального окружения venv. 
- Команда: `pip install -r requirements.txt`
- Результат: успех
 
### Отчёт о запуске
- Команда: `python run.py`
- Результат: успех

- Команда: `python --version`
- Результат: Python 3.9.13

### Установка дополнительных библиотек
Установлены notebook, matplotlib, seaborn, pytest. Нужны для собственного EDA и тестов; в исходном проекте их нет

### Версии ключевых библиотек

Python 3.9.13, pip 23.2.1. Полный список — `pip list` / `requirements.txt`.

| Библиотека | Версия | Назначение | Источник версии |
|---|---|---|---|
| dash | 2.1.0 | веб-приложение | Pipfile.lock |
| dash-bootstrap-components | 1.0.3 | вёрстка приложения | Pipfile.lock |
| plotly | 5.6.0 | графики в приложении | закреплена вручную |
| flask | 2.0.3 | сервер Dash | Pipfile.lock |
| werkzeug | 2.0.3 | зависимость Flask | закреплена вручную |
| pandas | 1.3.5 | работа с таблицами | Pipfile.lock |
| numpy | 1.21.6 | вычисления | закреплена вручную |
| scipy | 1.7.3 | зависимость statsmodels | закреплена вручную |
| statsmodels | 0.13.2 | модель OLS | закреплена вручную |
| gunicorn | 20.1.0 | сервер для развёртывания | Pipfile.lock |
| notebook | 7.5.7 | блокнот Jupyter | добавлено для EDA |
| matplotlib | 3.8.4 | графики | добавлено для EDA |
| seaborn | 0.13.2 | статистические графики | добавлено для EDA |
| pytest | 8.4.2 | тесты | добавлено для EDA |

<details>
<summary>Прочие установленные пакеты</summary>

| Пакет | Версия |
|---|---|
| Brotli | 1.0.9 |
| click | 8.0.3 |
| colorama | 0.4.4 |
| dash-core-components | 2.0.0 |
| dash-html-components | 2.0.0 |
| dash-table | 5.0.0 |
| Flask-Compress | 1.10.1 |
| itsdangerous | 2.0.1 |
| Jinja2 | 3.0.3 |
| MarkupSafe | 2.0.1 |
| packaging | 21.3 |
| patsy | 0.5.2 |
| pyparsing | 3.3.3 |
| python-dateutil | 2.9.0.post0 |
| pytz | 2026.4 |
| six | 1.17.0 |
| tenacity | 9.1.2 |
| setuptools | 68.2.0 |
| wheel | 0.41.2 |

</details>

### 