### Week 17 : 27 Oct 25

Precision : What is the % that is correct identified
Recall: After Predicting recall to see if what i have identified is correctly identified
F1 Score : Balance between Precision and Recall

### Formulas

### Legends

- Prescision : P
- Recall : R
- True Possitive : TP
- True Negative: TN
- False Possitive : FP
- False Negative : FN

P = TP / TP + FP

R = TP / TP + FN

F1 Score = 2 X ( (P \* R) / (P + R))

### Example of TP TN FP FN

| Actual \ Predicted | Spam                   | Not Spam               |
| ------------------ | ---------------------- | ---------------------- |
| **Spam**           | ✅ True Positive (TP)  | ❌ False Negative (FN) |
| **Not Spam**       | ❌ False Positive (FP) | ✅ True Negative (TN)  |

High precision = few false positives [Able to guess it correctly]
High recall = few false negatives [Revalidating that what is guessed is correct]

---

### 30th Oct 25

### Naive Bayes Classification Algorithm

- Used in whether email is spam or not
- Identifying this or that
- Results are fast and accurate as it works on probablity
- **P(B|A)** = (P(B) X P(A|B))/P(A)
- ***
