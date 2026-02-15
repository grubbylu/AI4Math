# Testing Custom Problems on Kaggle

## Steps

### 1. Upload `custom_test_problems.csv` to Kaggle
- Go to your Kaggle notebook
- Upload `custom_test_problems.csv` as a dataset or place it in the working directory

### 2. Modify the CSV loading cell (cell after solver init)
Replace the reference CSV loading with:

```python
# Load custom test problems
df = pd.read_csv("custom_test_problems.csv")

# Store ground truth answers for accuracy calculation
ground_truth = dict(zip(df["id"], df["answer"])) if "answer" in df.columns else {}

# Create input file without answers
df.drop("answer", axis=1, errors="ignore").to_csv("custom_test.csv", index=False)

predictions = {}
correct_count = 0
total_count = 0
```

### 3. Modify the gateway cell
Change:
```python
inference_server.run_local_gateway(("reference.csv",))
```
To:
```python
inference_server.run_local_gateway(("custom_test.csv",))
```

### 4. Run all cells and observe results

## Test Problems Overview

| ID | Topic | What It Tests |
|----|-------|---------------|
| test_01 | Divisor sums | Finding smallest integers with divisor pair constraints (similar to Problem #7's "n-Norwegian") |
| test_02 | Divisor counting | Computing d(k^2) sum over large range — needs formula, not brute force |
| test_03 | Divisor sum ratio | Floor of sigma(n)/n — filtering over large range with number-theoretic insight |
| test_04 | Prime factorization | Counting coprime factor pairs of primorial — tests combinatorial number theory |
| test_05 | Abundant numbers | Smallest 2-abundant number — straightforward but tests divisor sum computation |

## What to Look For
- Does the model use analytical formulas or try brute force?
- Does it correctly handle modular arithmetic?
- Does it verify its answers?
- How many Python calls does it make?
- Does the adaptive retry trigger on any problems?
