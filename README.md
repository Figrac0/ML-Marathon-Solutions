#  Holiday Refrigerator Recommendations Challenge

<div align="center">
  <a href="https://colab.research.google.com/drive/1G-VFIDoccPqYCqXMFzVjFP-8u2Lmm56-?usp=sharing" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab">
    <br>
    <strong>Click to open in Google Colab</strong>
  </a>
</div>
---

<p align="center">
  <img src="https://github.com/Figrac0/ML-Marathon-Solutions/blob/Second-Task/img/1.png" width="290" height="260"/>
</p>

**Figure 1: Project File Structure**  
Shows the complete directory structure including initial datasets (`train_holodilnik.csv`, `test_holodilnik.csv`, `dishes.csv`, `users.csv`), submission files from different model versions, and intermediate outputs.

<p align="center">
  <img src="https://github.com/Figrac0/ML-Marathon-Solutions/blob/Second-Task/img/2.png" width="640" height="290"/>
</p>

**Figure 2: CatBoostRanker Training Logs (Full Model)**  
Displays the detailed training progression of the main CatBoostRanker model over 1,200 iterations. Shows NDCG@5 improvement from 0.390 to 0.514, with early stopping triggered at iteration 1,199. Training completed in 2 hours 15 minutes.

<p align="center">
  <img src="https://github.com/Figrac0/ML-Marathon-Solutions/blob/Second-Task/img/3.png" width="620" height="180"/>
</p>

**Figure 3: Optimized Model Training Logs**  
Shows the faster training of the optimized CatBoostRanker with bitmask operations. Achieved NDCG@5 of 0.508 in just 499 iterations (1 hour 5 minutes), demonstrating efficient computation while maintaining competitive performance.

##  Challenge Overview
After the holiday season, a family refrigerator is still full of various dishes, but family members keep changing their minds about what they want to eat. The goal is to predict which dish a person will choose based on the context and available dishes, providing top-5 personalized recommendations with **NDCG@5** as the evaluation metric.

**Key constraints:**
- Recommend exactly 5 different dish IDs per query
- All recommendations must be from the available candidates (20 dishes per query)
- Target dish is always among the candidates

##  Solution Architecture

###  **Data Validation & Preprocessing**
- Comprehensive schema validation for all input files (`dishes.csv`, `users.csv`, `train.csv`, `test.csv`)
- Candidate integrity checks (ensuring target dish is among candidates)
- Reference integrity between tables
- Memory optimization with downcasting (int8/int16, float32)

###  **Feature Engineering**

#### **User-Dish Interaction Features:**
- **Allergen conflict detection**: Binary flag if user allergies intersect with dish allergens
- **Tag matching**: Count of overlapping tags between user preferences and dish tags
- **Category preferences**: Dessert/drink affinity based on user flags (sweet_tooth, coffee_addict)
- **Calorie sensitivity**: Penalties for users on diet or preferring light food

#### **Contextual Features:**
- Temporal features: `day`, `meal_slot` (breakfast/lunch/dinner/late_snack)
- Social context: `guests_count`, `hangover_level`
- Refrigerator state: `fridge_load_pct`
- Positional feature: `candidate_pos` (order in which dish appears in candidate list)

###  **Modeling Approaches**

#### **1. Baseline Model (Global Popularity)**
- Simple yet effective: recommends most globally popular dishes among available candidates
- **Holdout NDCG@5**: 0.278
- Serves as a strong benchmark for comparison

#### **2. CatBoostRanker with Comprehensive Features**
- **Architecture**: Pairwise ranking model using YetiRankPairwise loss
- **Training**: 1,200 iterations with early stopping (patience=100)
- **Features**: 24 engineered features including user-dish interactions
- **Holdout NDCG@5**: **0.514** (85% improvement over baseline)
- **Training time**: ~2 hours 15 minutes

#### **3. Optimized CatBoostRanker with Bitmask Operations**
- **Optimization**: Used bitmask operations for faster tag/allergen overlap computation
- **Performance**: Reduced training time while maintaining quality
- **Holdout NDCG@5**: 0.508
- **Training time**: ~1 hour 5 minutes

###  **Key Technical Decisions**

1. **Candidate-Level Dataset**: Transformed from query-level to candidate-level format (3M training samples)
2. **Negative Sampling**: 9 negative samples per positive for training efficiency
3. **Group-Based Validation**: Split by query groups to prevent data leakage
4. **Strict Output Validation**: Ensured all recommendations are unique and within candidate sets
5. **Feature Importance**: Most impactful features were allergen conflicts and tag overlaps

## 📈 Results Summary

| Model | Holdout NDCG@5 | Training Time | Key Features |
|-------|----------------|---------------|--------------|
| Baseline (Popularity) | 0.278 | < 1 min | Global dish popularity |
| CatBoostRanker (Full) | **0.514** | ~2h 15m | 24 engineered features |
| CatBoostRanker (Optimized) | 0.508 | ~1h 5m | Bitmask operations |


## 🔧 Technical Implementation

### **Block 1–2: Data Loading & Validation**
**Purpose:** Initialize the environment and perform comprehensive data quality checks to ensure downstream processing reliability.

**Key Operations:**
- **Robust file discovery** - Handles multiple naming conventions (e.g., `train.csv` vs `train_holodilnik.csv`)
- **Schema validation** - Verifies all required columns exist in each dataset using strict assertions
- **Data integrity checks:**
  - Uniqueness validation for primary keys (`dish_id`, `user_id`, `query_id`)
  - Candidate value range validation (1-200)
  - Reference integrity between tables (foreign key constraints)
  - Target dish presence in candidate lists (100% validation)
- **Memory-efficient loading** with proper data types

**Output:** Validated datasets ready for feature engineering with zero missing values or structural inconsistencies.

### **Block 3–4: Baseline Model & Evaluation Framework**
**Purpose:** Establish a performance benchmark and implement the core evaluation metric (NDCG@5).

**Key Components:**
- **NDCG@5 implementation** - Custom metric calculation matching competition requirements
- **Global popularity baseline** - Recommends most frequently chosen dishes from available candidates
- **Holdout validation** - 20% random split for unbiased performance estimation
- **Submission validation** - Ensures recommendations are unique and within candidate sets

**Results:** Baseline NDCG@5 = 0.278, providing a reference point for advanced models.

### **Block 5: Advanced CatBoostRanker with Feature Engineering**
**Purpose:** Build a production-ready ranking system with sophisticated feature engineering.

#### **Phase 1: Data Optimization**
- **Memory downcasting** - Reduces memory usage by 60-80% via uint8/int16 types
- **String normalization** - Standardizes categorical/text fields with consistent NA handling
- **Lookup dictionaries** - Precomputes dish/user attributes for O(1) access during feature generation

#### **Phase 2: Feature Engineering Pipeline**
**User-Dish Interaction Features:**
- `allergen_conflict` - Binary flag for allergic reactions
- `liked/disliked_overlap` - Count of matching tags between user preferences and dish characteristics
- `is_dessert/is_drink` - Category indicators for preference modeling

**Contextual Features:**
- Temporal: `day`, `meal_slot` (categorical)
- Social: `hangover_level`, `guests_count`
- Behavioral: `diet_mode`, `fridge_load_pct`
- Positional: `candidate_pos` (order bias in candidate list)

**Derived Features:**
- `diet_cal_penalty` - Interaction between diet mode and calorie content
- `sweet_dessert_aff` - Sweet tooth preference combined with dessert category
- `coffee_drink_aff` - Coffee addiction signal with drink category

#### **Phase 3: Model Architecture**
- **Algorithm:** CatBoostRanker with YetiRankPairwise loss (optimized for ranking tasks)
- **Training Strategy:** Group-based holdout (by query_id) to prevent data leakage
- **Hyperparameters:** 1200 iterations, learning_rate=0.08, depth=6, L2 regularization
- **Early Stopping:** 100 iteration patience based on validation NDCG@5

#### **Phase 4: Inference & Validation**
- **Candidate-level scoring** - Each dish receives a relevance score per query
- **Top-5 selection** - Greedy selection ensuring uniqueness and candidate validity
- **Strict validation** - Verifies no duplicates or out-of-candidate recommendations

**Performance:** NDCG@5 = 0.514 (85% improvement over baseline)

### **Block 6: Optimized Implementation with Bitmask Operations**
**Purpose:** Improve computational efficiency while maintaining model quality.

**Optimizations:**
- **Bitmask encoding** - Represents tag/allergen sets as integer bitmasks
- **Fast set operations** - Uses `bit_count()` and bitwise `&` for overlap computation
- **Reduced training time** - 1 hour 5 minutes vs 2 hours 15 minutes (47% faster)
- **Streamlined features** - Maintains core feature set with optimized computation

**Performance:** NDCG@5 = 0.508 (slight trade-off for 2x speed improvement)

## 📊 Model Comparison Summary

| Model | NDCG@5 | Training Time | Key Innovation |
|-------|--------|---------------|----------------|
| Baseline (Popularity) | 0.278 | <1 min | Simple global statistics |
| CatBoostRanker (Full) | **0.514** | 2h 15m | Comprehensive feature engineering |
| CatBoostRanker (Optimized) | 0.508 | 1h 5m | Bitmask operations for efficiency |

## 🎯 Key Technical Insights

1. **Feature Importance:** Allergen conflicts and tag overlaps were most predictive
2. **Data Leakage Prevention:** Group-based validation was crucial for reliable estimates
3. **Memory Management:** Downcasting reduced memory usage from ~2GB to ~400MB
4. **Production Readiness:** All submissions pass strict validation (100% valid recommendations)
5. **Scalability:** Bitmask approach shows 2x speed improvement with minimal quality loss

## 🚀 Production Deployment Features

- **Robust error handling** for all edge cases
- **Memory-efficient processing** suitable for large datasets
- **Modular design** allowing easy feature addition/removal
- **Complete reproducibility** with fixed random seeds
- **Comprehensive logging** for debugging and monitoring
