# Wine Multiclass Classification with PyTorch

sklearn `load_wine` 데이터셋(N=178, 13 features, 3 classes)을 사용한 MLP 다중 분류 실습. ver0 baseline의 두 가지 학습 실패 원인을 진단하고 ver1에서 수정한 사례.

---

## 1. 결과 요약 (단일 split, `random_state=2024`)

| Version | Test Accuracy | 주요 변경 |
|---------|---------------|-----------|
| ver0    | ≈ 0.36        | baseline (스케일링 없음, 출력층 Softmax + `CrossEntropyLoss` 중복) |
| ver1    | ≈ 0.97        | `StandardScaler` 적용 + 출력층 Softmax 제거 |

> ⚠️ 위 수치는 단일 train/val/test 분할(`random_state=2024`)의 결과이며, test set 크기가 약 36 샘플에 불과하다. 따라서, 일반화 성능을 단정하기에는 표본이 작다는 한계가 있다.

---

## 2. 진단: ver0의 두 가지 문제점

### 2.1. 입력 피처 스케일 불일치

Wine 데이터셋 13개 피처는 단위 및 분포 차이가 큼.

| 피처 예시 | 대략적 범위 |
|-----------|-------------|
| `proline` | [278, 1680] |
| `magnesium` | [70, 162] |
| `nonflavanoid_phenols` | [0.13, 0.66] |

- 스케일이 큰 피처가 첫 선형 계층의 활성값(pre-activation)을 지배함.
- Adam이 parameter-wise adaptive learning rate를 제공하지만, ReLU 활성과 결합 시 일부 뉴런의 dead/saturation 위험이 증가함.

### 2.2. Softmax + CrossEntropyLoss 중복 적용

- `torch.nn.CrossEntropyLoss`는 내부적으로 `LogSoftmax`와 `NLLLoss`를 결합한 연산이며, 입력으로 **raw logits**를 기대함 (PyTorch 공식 문서 기준).
- ver0의 forward()는 출력 직전에 `nn.Softmax(dim=1)`을 한 번 더 적용 → 손실은 사실상 `-log(softmax(softmax(logits)))`를 계산.
- 이중 Softmax는 출력 분포를 평탄화하여 그래디언트 신호를 극도로 약화시키고, 결과적으로 다수 클래스로의 mode collapse를 유발할 수 있음.

두 문제는 독립적이지만, **결합 시 학습이 사실상 정체됨**. ver0의 test set에서 모델이 단일 클래스에만 예측을 집중한 것은 이 가설과 일치함.

---

## 3. 수정 내역 (ver0 → ver1)

```python
# (1) StandardScaler: train에 fit, val/test에는 transform만 적용 (leakage 방지)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val   = scaler.transform(x_val)
x_test  = scaler.transform(x_test)

# (2) 출력층 Softmax 제거: CrossEntropyLoss에 raw logits 전달
def forward(self, x):
    x = self.relu(self.linear_1(x))
    x = self.relu(self.linear_2(x))
    x = self.relu(self.linear_3(x))
    x = self.linear_4(x)            # softmax 제거
    return x
```

기타 하이퍼파라미터(epochs=100, batch_size=16, lr=1e-3, Adam, 4-layer MLP 구조)는 ver0와 동일하게 유지하여 변경 요인을 위 두 가지로 한정.

---

## 4. 한계 및 추가 검증 제안

이 단일 검증에서는 향상된 정확도 (97%)와 좋은 재현율(Class 0 = 1.00, Class 2 = 0.92)을 보였다.
그러나, 본 결과의 일반화를 주장하기에 앞서, 아래 항목들의 보완 검증이 필요함.

1. **단일 split 의존성**: 결과는 단일 `random_state=2024`에 한정됨. K-fold CV(예: stratified 5-fold) 또는 5–10개 seed에 대한 평균 ± std 보고 권장.
2. **테스트 표본 크기**: N_test ≈ 36이므로 1샘플 오분류 = 약 2.8%p 변동. 정확도의 신뢰 구간이 매우 넓음.
3. **시드 고정 누락**: 두 버전 모두 `torch.manual_seed` / `numpy.random.seed`를 설정하지 않음. 완전한 재현을 위해서는 명시적 시드 고정 및 cuDNN deterministic 설정 필요.
4. **모델 구조 적정성 미검증**: 13→32→64→128→3 구조는 N_train ≈ 113에 비해 파라미터가 과다할 가능성. 더 단순한 baseline(예: 1–2 hidden layer) 및 정규화(Dropout, weight decay) 대조 미실시.
5. **Stratified split 미적용**: 클래스 분포(class 0: 59, class 1: 71, class 2: 48)는 비교적 균형이나, split 시 stratify 옵션 미사용으로 fold별 분포 편차 가능성 존재.
6. **One-hot label 사용**: ver1도 여전히 one-hot encoded label을 `CrossEntropyLoss`에 전달. PyTorch ≥ 1.10은 soft label을 지원하므로 결과상 차이는 없으나, 관례상 class index를 직접 전달하는 것이 표준이며 코드도 단순해짐.
7. **비교 baseline 부재**: 본 데이터 규모에서는 로지스틱 회귀, 랜덤 포레스트 등 고전 모델이 종종 동등 이상의 성능을 보임. MLP의 우위 주장을 위해서는 동일 split에서의 sklearn baseline 비교가 필요함.

---

## 5. 파일 구성

```
.
├── wine_multiclass_classification_pytorch_ver0.py   # baseline (학습 실패 사례)
├── wine_multiclass_classification_pytorch_ver1.py   # scaling + softmax 제거 적용
└── README.md
```

## 6. 실행 환경

- Python 3.x
- PyTorch, scikit-learn
- 재현 시드: `random_state=2024` (sklearn split에 한정; PyTorch 측 시드 미고정 — §4의 3번 항목 참고)
