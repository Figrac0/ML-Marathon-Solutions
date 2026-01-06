# Task 1: Christmas Tree Survival Prediction

<div align="center">
  <a href="https://colab.research.google.com/drive/1V_3PW32s4i6m9AQcWBq-M5eouJZ9zA3A?usp=sharing" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab">
    <br>
    <strong>Click to open in Google Colab</strong>
  </a>
</div>

<p align="center">
  <img src="https://github.com/Figrac0/ML-Marathon-Solutions/blob/main/img/1.png" width="230" height="380"/>
</p>

**Figure 1: Project File Structure**  
This image shows the complete directory structure of the ML project, including training/test datasets, submission files, logs, and model outputs. Key files include `train.csv`, `test.csv`, submission files with different model configurations, and validation logs.

<p align="center">
  <img src="https://github.com/Figrac0/ML-Marathon-Solutions/blob/main/img/2.png" width="450" height="600"/>
</p>

**Figure 2: CatBoost Cross-Validation Results**  
This image displays the detailed CatBoost training logs from 5-fold cross-validation. Each fold shows the progressive improvement of AUC scores, best iterations, early stopping triggers due to overfitting detection, and the final ensemble predictions for sample apartments.

## Problem Statement

The task is to predict the probability that a Christmas tree in an apartment will survive until January 18th. The prediction is based on apartment characteristics and tree maintenance conditions. The dataset contains information from 30,000 apartments for training and 18,000 apartments for testing.

### Objective
Predict probability `survived_to_18jan = 1` (float between 0.0 and 1.0) for each apartment in the test set.

### Evaluation Metric
ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

## Dataset Overview

### Data Dimensions
- Training set: 30,000 samples × 30 features
- Test set: 18,000 samples × 29 features (no target)
- Target variable: Binary (0 = did not survive, 1 = survived)

### Key Features
- **Apartment characteristics**: building age, wing, floor, area, ceiling height, window quality, heating type
- **Environmental factors**: room temperature, humidity, ventilation frequency, radiator distance
- **Household factors**: children count, cat presence, robot vacuum
- **Tree characteristics**: species, height, form, stand type, cutting date, potting status
- **Maintenance factors**: watering frequency, mist spraying, garland type and hours, ornaments weight

### Data Quality
- Target distribution: 48.06% survived, 51.94% did not survive
- Missing values present (5-7% in humidity, watering frequency, ornaments weight, garland hours)
- Categorical features: 8 columns (wing, window_quality, heating_type, tree_species, tree_form, stand_type, tinsel_level)

## Methodology

### 1. Baseline Modeling
- **Algorithm**: CatBoostClassifier (gradient boosting optimized for categorical data)
- **Validation**: 5-fold Stratified Cross-Validation
- **Configuration**:
  - Iterations: 6000
  - Learning rate: 0.03
  - Depth: 7
  - L2 regularization: 4.0
  - Early stopping with 300 iterations patience

### 2. Model Comparison
Tested four different configurations:
- **A_baseline**: Standard configuration (best performer)
- **C_deeper_reg**: Deeper trees with stronger regularization
- **D_shallower_fast**: Shallower trees with faster learning
- **E_balanced**: Balanced configuration with Bayesian bootstrapping

### 3. Ensemble Approach
- **Linear blending**: Weighted combination of Model A and Model B
- **Optimal weight search**: Grid search over 101 weights (0.0 to 1.0)
- **Best blend**: 77% Model A + 23% Model B

### 4. Feature Engineering
Created 17 new features:
- **Binning**: Floor, apartment area, temperature, humidity, tree height, garland hours, cutting days, watering frequency
- **Categorical interactions**: Wing × window quality, heating × window quality, stand type × potted tree, species × form, tinsel × garland, cat × children
- **Numerical transformations**: Ornaments weight per height, inverse radiator distance, temperature minus humidity ratio

## Results

### Cross-Validation Performance

| Model | OOF AUC | Mean Fold AUC | Std Dev |
|-------|---------|---------------|---------|
| Baseline CatBoost | 0.67202 | 0.67212 | 0.00331 |
| Model B (Regularized) | 0.67038 | 0.67052 | 0.00311 |
| Best Blend (77%A + 23%B) | 0.67211 | - | - |
| With Feature Engineering | 0.67271 | 0.67282 | 0.00323 |

### Key Findings
1. Feature engineering provided a modest but consistent improvement (+0.00069 AUC)
2. The baseline configuration performed best among individual models
3. Ensemble blending offered marginal improvement over single models
4. Model stability was good with standard deviation < 0.0035 across folds

## Technical Implementation

### Dependencies
- Python 3.7+
- catboost 1.2.8
- pandas 2.2.2
- numpy 1.26.4
- scikit-learn 1.4.2

### Code Structure
1. **Data Loading & Validation**: Load datasets and verify integrity
2. **Preprocessing**: Handle categorical features and missing values
3. **Cross-Validation Setup**: 5-fold stratified splitting
4. **Model Training**: CatBoost with early stopping
5. **Prediction & Submission**: Generate probability predictions and format output

### Validation Strategy
- Fixed random seed (42) for reproducibility
- Stratified K-Fold to preserve class distribution
- Out-of-Fold predictions for reliable performance estimation
- Assertions to ensure data integrity throughout pipeline

## Files

### Input Data
- `train.csv`: Training data with target
- `test.csv`: Test data without target
- `sample_submission.csv`: Submission format template

### Output Files
- `submission_catboost_baseline.csv`: Baseline model predictions
- `submission_best_blend.csv`: Ensemble model predictions
- `submission_best_single_model.csv`: Best single model predictions
- `submission_catboost_stable_fe.csv`: Best overall predictions with feature engineering

### Code Files
- `Codemrock_1.ipynb`: Jupyter notebook with complete analysis
- `codemrock_1.py`: Python script version

- ### Current Limitations

1. **Feature Engineering Scope**  
   Feature engineering was applied globally across the entire dataset rather than within cross-validation folds. This approach may introduce subtle data leakage, as feature transformations use information from both training and validation sets during cross-validation.

2. **Hyperparameter Optimization**  
   Limited exploration of hyperparameter space. The current configuration was selected based on initial experimentation rather than systematic optimization, potentially leaving performance improvements unexplored.

3. **Algorithm Diversity**  
   The solution relies exclusively on CatBoost algorithm. While CatBoost is well-suited for tabular data with categorical features, ensemble approaches combining multiple algorithms could provide better generalization and robustness.

4. **Missing Value Handling**  
   Missing values are handled implicitly by CatBoost's internal mechanisms. More sophisticated imputation strategies (multiple imputation, predictive modeling of missing values) were not explored.

5. **Categorical Feature Encoding**  
   Categorical features are processed using CatBoost's native handling. Alternative encoding methods (target encoding, frequency encoding) with proper cross-validation were not implemented.

## Постановка задачи

Задача состоит в прогнозировании вероятности того, что новогодняя ёлка в квартире доживёт до 18 января. Прогноз основан на характеристиках квартиры и условиях содержания ёлки. Набор данных содержит информацию о 30 000 квартир для обучения и 18 000 квартир для тестирования.

### Цель
Предсказать вероятность `survived_to_18jan = 1` (число с плавающей точкой от 0.0 до 1.0) для каждой квартиры в тестовом наборе.

### Метрика оценки
ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

## Обзор данных

### Размеры данных
- Обучающая выборка: 30 000 примеров × 30 признаков
- Тестовая выборка: 18 000 примеров × 29 признаков (без целевой переменной)
- Целевая переменная: Бинарная (0 = не дожила, 1 = дожила)

### Ключевые признаки
- **Характеристики квартиры**: возраст дома, сторона дома, этаж, площадь, высота потолков, качество окон, тип отопления
- **Факторы окружающей среды**: температура в комнате, влажность, частота проветривания, расстояние до радиатора
- **Домохозяйственные факторы**: количество детей, наличие кота, наличие робота-пылесоса
- **Характеристики ёлки**: порода, высота, форма, тип подставки, дата срубки, наличие горшка
- **Факторы ухода**: частота полива, опрыскивание водой, тип гирлянды и время работы, вес украшений

### Качество данных
- Распределение целевой переменной: 48.06% дожили, 51.94% не дожили
- Присутствуют пропущенные значения (5-7% в влажности, частоте полива, весе украшений, времени работы гирлянды)
- Категориальные признаки: 8 столбцов (сторона дома, качество окон, тип отопления, порода ёлки, форма ёлки, тип подставки, уровень мишуры)

## Методология

### 1. Базовое моделирование
- **Алгоритм**: CatBoostClassifier (градиентный бустинг, оптимизированный для категориальных данных)
- **Валидация**: 5-кратная стратифицированная кросс-валидация
- **Конфигурация**:
  - Количество итераций: 6000
  - Скорость обучения: 0.03
  - Глубина деревьев: 7
  - L2-регуляризация: 4.0
  - Ранняя остановка с терпением 300 итераций

### 2. Сравнение моделей
Протестированы четыре различные конфигурации:
- **A_baseline**: Стандартная конфигурация (лучший результат)
- **C_deeper_reg**: Более глубокие деревья с усиленной регуляризацией
- **D_shallower_fast**: Более мелкие деревья с быстрым обучением
- **E_balanced**: Сбалансированная конфигурация с байесовским бутстраппингом

### 3. Ансамблевый подход
- **Линейное блендирование**: Взвешенная комбинация Модели A и Модели B
- **Поиск оптимальных весов**: Поиск по сетке из 101 веса (от 0.0 до 1.0)
- **Лучший бленд**: 77% Модель A + 23% Модель B

### 4. Конструирование признаков
Создано 17 новых признаков:
- **Биннинг**: Этаж, площадь квартиры, температура, влажность, высота ёлки, время работы гирлянды, дни до срубки, частота полива
- **Взаимодействия категориальных признаков**: Сторона дома × качество окон, отопление × качество окон, тип подставки × наличие горшка, порода × форма, уровень мишуры × тип гирлянды, наличие кота × количество детей
- **Числовые преобразования**: Вес украшений на высоту, обратное расстояние до радиатора, температура минус отношение влажности

## Результаты

### Производительность кросс-валидации

| Модель | OOF AUC | Средний AUC фолдов | Стандартное отклонение |
|--------|---------|--------------------|------------------------|
| Базовый CatBoost | 0.67202 | 0.67212 | 0.00331 |
| Модель B (регуляризованная) | 0.67038 | 0.67052 | 0.00311 |
| Лучший бленд (77%A + 23%B) | 0.67211 | - | - |
| С конструированием признаков | 0.67271 | 0.67282 | 0.00323 |

### Ключевые выводы
1. Конструирование признаков дало скромное, но стабильное улучшение (+0.00069 AUC)
2. Базовая конфигурация показала лучший результат среди отдельных моделей
3. Ансамблевое блендирование дало незначительное улучшение по сравнению с одиночными моделями
4. Стабильность модели была хорошей со стандартным отклонением < 0.0035 между фолдами

## Техническая реализация

### Зависимости
- Python 3.7+
- catboost 1.2.8
- pandas 2.2.2
- numpy 1.26.4
- scikit-learn 1.4.2

### Структура кода
1. **Загрузка и валидация данных**: Загрузка наборов данных и проверка целостности
2. **Предобработка**: Обработка категориальных признаков и пропущенных значений
3. **Настройка кросс-валидации**: 5-кратное стратифицированное разбиение
4. **Обучение модели**: CatBoost с ранней остановкой
5. **Предсказание и сабмит**: Генерация вероятностных предсказаний и форматирование вывода

### Стратегия валидации
- Фиксированный random seed (42) для воспроизводимости
- Стратифицированная K-Fold для сохранения распределения классов
- Out-of-Fold предсказания для надежной оценки производительности
- Утверждения (assert) для обеспечения целостности данных на протяжении всего пайплайна

## Файлы

### Входные данные
- `train.csv`: Обучающие данные с целевой переменной
- `test.csv`: Тестовые данные без целевой переменной
- `sample_submission.csv`: Шаблон формата сабмита

### Выходные файлы
- `submission_catboost_baseline.csv`: Предсказания базовой модели
- `submission_best_blend.csv`: Предсказания ансамблевой модели
- `submission_best_single_model.csv`: Предсказания лучшей одиночной модели
- `submission_catboost_stable_fe.csv`: Лучшие предсказания с конструированием признаков

### Файлы кода
- `Codemrock_1.ipynb`: Jupyter ноутбук с полным анализом
- `codemrock_1.py`: Версия на Python

## Текущие ограничения

1. **Область конструирования признаков**  
   Конструирование признаков применялось глобально ко всему набору данных, а не внутри фолдов кросс-валидации. Этот подход может создавать незначительную утечку данных, так как преобразования признаков используют информацию как из обучающих, так и из валидационных наборов во время кросс-валидации.

2. **Оптимизация гиперпараметров**  
   Ограниченное исследование пространства гиперпараметров. Текущая конфигурация была выбрана на основе начальных экспериментов, а не систематической оптимизации, что могло оставить неисследованными потенциальные улучшения производительности.

3. **Разнообразие алгоритмов**  
   Решение полагается исключительно на алгоритм CatBoost. Хотя CatBoost хорошо подходит для табличных данных с категориальными признаками, ансамблевые подходы, сочетающие несколько алгоритмов, могли бы обеспечить лучшую обобщающую способность и устойчивость.

4. **Обработка пропущенных значений**  
   Пропущенные значения обрабатываются неявно внутренними механизмами CatBoost. Более сложные стратегии импутации (множественная импутация, прогнозное моделирование пропущенных значений) не исследовались.

5. **Кодирование категориальных признаков**  
   Категориальные признаки обрабатываются с использованием нативного механизма CatBoost. Альтернативные методы кодирования (таргет-кодирование, частотное кодирование) с правильной кросс-валидацией не были реализованы.
