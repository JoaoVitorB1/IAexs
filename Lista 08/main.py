from __future__ import annotations
import math
import random
from typing import List, Tuple, Dict

SEED = 123
random.seed(SEED)

LR = 0.5
EPOCHS_XOR = 8000
EPOCHS_7SEG = 10000
NOISE_TRAIN_P = 0.05 

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def dsigmoid_from_y(y: float) -> float:
    return y * (1.0 - y)

def relu(x: float) -> float:
    return x if x > 0.0 else 0.0

def drelu_from_z(z: float) -> float:
    return 1.0 if z > 0.0 else 0.0

# XOR
X_XOR: List[List[float]] = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
Y_XOR: List[List[float]] = [[0.0], [1.0], [1.0], [1.0]]

# Referência clássica:
SEG7 = {
    0: [1,1,1,1,1,1,0],
    1: [0,1,1,0,0,0,0],
    2: [1,1,0,1,1,0,1],
    3: [1,1,1,1,0,0,1],
    4: [0,1,1,0,0,1,1],
    5: [1,0,1,1,0,1,1],
    6: [1,0,1,1,1,1,1],
    7: [1,1,1,0,0,0,0],
    8: [1,1,1,1,1,1,1],
    9: [1,1,1,1,0,1,1],
}

X_7: List[List[float]] = [list(map(float, SEG7[d])) for d in range(10)]

def enc_bits4(d: int) -> List[float]:
    return [float((d >> k) & 1) for k in (3,2,1,0)]

Y_7_bits: List[List[float]] = [enc_bits4(d) for d in range(10)]

def enc_onehot10(d: int) -> List[float]:
    v = [0.0]*10
    v[d] = 1.0
    return v

Y_7_onehot: List[List[float]] = [enc_onehot10(d) for d in range(10)]

def noisy_copy(x: List[float], p: float) -> List[float]:
    if p <= 0.0:
        return x[:]
    y = []
    for val in x:
        bit = 1 if val >= 0.5 else 0
        if random.random() < p:
            bit ^= 1  # flip
        y.append(float(bit))
    return y

class MLP:
    def __init__(self, sizes: List[int], seed: int = 123, hidden_act: str = "sigmoid", output_act: str = "sigmoid"):
        assert len(sizes) == 3, "Use [n_in, n_hidden, n_out]"
        self.sizes = sizes
        self.hidden_act = hidden_act
        self.output_act = output_act
        rnd = random.Random(seed)
        n_in, n_h, n_out = sizes
        def r():
            # pequena escala
            return rnd.uniform(-1.0, 1.0) * 0.5
        self.w1 = [[r() for _ in range(n_in)] for _ in range(n_h)]
        self.b1 = [r() for _ in range(n_h)]
        self.w2 = [[r() for _ in range(n_h)] for _ in range(n_out)]
        self.b2 = [r() for _ in range(n_out)]

    def forward(self, x: List[float]):
        # camada oculta
        z1 = []
        for j in range(len(self.b1)):
            s = self.b1[j]
            for i, xi in enumerate(x):
                s += self.w1[j][i] * xi
            z1.append(s)
        if self.hidden_act == "sigmoid":
            a1 = [sigmoid(z) for z in z1]
        elif self.hidden_act == "relu":
            a1 = [relu(z) for z in z1]
        else:
            raise ValueError("hidden_act deve ser 'sigmoid' ou 'relu'")

        z2 = []
        for k in range(len(self.b2)):
            s = self.b2[k]
            for j, aj in enumerate(a1):
                s += self.w2[k][j] * aj
            z2.append(s)
        if self.output_act == "sigmoid":
            a2 = [sigmoid(z) for z in z2]
        else:
            raise ValueError("output_act deve ser 'sigmoid' neste script")
        return x, z1, a1, z2, a2

    def backward(self, cache, target: List[float], lr: float):
        x, z1, a1, z2, y = cache
        delta2 = []
        for k in range(len(y)):
            delta2.append((y[k] - target[k]) * dsigmoid_from_y(y[k]))
        delta1 = []
        for j in range(len(z1)):
            s = 0.0
            for k in range(len(delta2)):
                s += self.w2[k][j] * delta2[k]
            if self.hidden_act == "sigmoid":
                s *= a1[j] * (1.0 - a1[j])
            else:
                s *= drelu_from_z(z1[j])
            delta1.append(s)
        for k in range(len(self.w2)):
            for j in range(len(self.w2[k])):
                self.w2[k][j] -= lr * (delta2[k] * a1[j])
            self.b2[k] -= lr * delta2[k]
        for j in range(len(self.w1)):
            for i in range(len(self.w1[j])):
                self.w1[j][i] -= lr * (delta1[j] * x[i])
            self.b1[j] -= lr * delta1[j]

    def mse(self, X: List[List[float]], Y: List[List[float]]) -> float:
        s = 0.0
        n = 0
        for x, t in zip(X, Y):
            _, _, _, _, y = self.forward(x)
            for k in range(len(t)):
                e = y[k] - t[k]
                s += 0.5 * e * e
                n += 1
        return s / max(1, n)

def train_epoch(model: MLP, X: List[List[float]], Y: List[List[float]], lr: float = LR, noise_p: float = 0.0):
    idx = list(range(len(X)))
    random.shuffle(idx)
    for ii in idx:
        x0 = X[ii]
        t = Y[ii]
        x = noisy_copy(x0, noise_p) if noise_p > 0.0 else x0
        cache = model.forward(x)
        model.backward(cache, t, lr)

def decode_bits4_to_digit(bits: List[float]) -> int:
    b = [1 if v >= 0.5 else 0 for v in bits]
    val = (b[0] << 3) | (b[1] << 2) | (b[2] << 1) | b[3]
    return val

def evaluate_bits4(model: MLP, X: List[List[float]]) -> Tuple[float, List[Tuple[int,int]]]:
    acertos = 0
    pairs = []
    for d, x in enumerate(X):
        _, _, _, _, y = model.forward(x)
        pred = decode_bits4_to_digit(y)
        pairs.append((d, pred))
        if pred == d:
            acertos += 1
    return acertos / len(X), pairs

def evaluate_onehot(model: MLP, X: List[List[float]]) -> Tuple[float, List[Tuple[int,int]]]:
    acertos = 0
    pairs = []
    for d, x in enumerate(X):
        _, _, _, _, y = model.forward(x)
        pred = max(range(len(y)), key=lambda k: y[k])
        pairs.append((d, pred))
        if pred == d:
            acertos += 1
    return acertos / len(X), pairs

def accuracy_with_noise_bits4(model: MLP, X: List[List[float]], trials: int = 50, flip_prob: float = 0.1) -> float:
    acertos = 0
    total = 0
    for _ in range(trials):
        for d, x in enumerate(X):
            xn = noisy_copy(x, flip_prob)
            _, _, _, _, y = model.forward(xn)
            pred = decode_bits4_to_digit(y)
            if pred == d:
                acertos += 1
            total += 1
    return acertos / max(1, total)

def accuracy_with_noise_onehot(model: MLP, X: List[List[float]], trials: int = 50, flip_prob: float = 0.1) -> float:
    acertos = 0
    total = 0
    for _ in range(trials):
        for d, x in enumerate(X):
            xn = noisy_copy(x, flip_prob)
            _, _, _, _, y = model.forward(xn)
            pred = max(range(len(y)), key=lambda k: y[k])
            if pred == d:
                acertos += 1
            total += 1
    return acertos / max(1, total)

def append_txt(path: str, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    mdl_xor = MLP([2, 2, 1], seed=42, hidden_act="sigmoid", output_act="sigmoid")
    last_ms = []
    for ep in range(EPOCHS_XOR):
        train_epoch(mdl_xor, X_XOR, Y_XOR, lr=LR, noise_p=0.0)
        if ep >= EPOCHS_XOR - 5:
            last_ms.append(mdl_xor.mse(X_XOR, Y_XOR))
    mse_xor = mdl_xor.mse(X_XOR, Y_XOR)
    print(f"XOR — MSE: {mse_xor:.6f}")
    for x, t in zip(X_XOR, Y_XOR):
        _, _, _, _, y = mdl_xor.forward(x)
        print(f"{[int(x[0]), int(x[1])]} -> {y[0]:.4f} (bin {1 if y[0] >= 0.5 else 0})")
    print(f"XOR — últimos MSEs: {[round(v,6) for v in last_ms]}")

    with open("relatorio_resultados.txt", "w", encoding="utf-8") as f:
        f.write("Relatório – Backpropagation do zero (XOR e 7 segmentos)\n")
        f.write("===============================================\n\n")
        f.write("Equações (sigmoid + MSE)\n")
        f.write("E = 1/2 * sum_k (y_k - t_k)^2\n")
        f.write("delta_k = (y_k - t_k) * y_k (1 - y_k)\n")
        f.write("delta_j = (sum_k w_{jk} * delta_k) * h_j (1 - h_j)\n")
        f.write("dE/dw^{(l)}_{j,i} = delta^{(l)}_j * a^{(l-1)}_i\n")
        f.write("dE/db^{(l)}_j = delta^{(l)}_j\n")
        f.write("Atualização: w <- w - eta * gradE; b <- b - eta * gradE\n\n")
        f.write(f"XOR — MSE: {mse_xor:.6f}\n")
        for x, t in zip(X_XOR, Y_XOR):
            _, _, _, _, y = mdl_xor.forward(x)
            f.write(f"{[int(x[0]), int(x[1])]} -> {y[0]:.4f} (bin {1 if y[0] >= 0.5 else 0})\n")
        f.write(f"XOR — últimos MSEs: {[round(v,6) for v in last_ms]}\n\n")

    mdl_7 = MLP([7, 5, 4], seed=777, hidden_act="sigmoid", output_act="sigmoid")
    for ep in range(EPOCHS_7SEG):
        train_epoch(mdl_7, X_7, Y_7_bits, lr=LR, noise_p=NOISE_TRAIN_P)
    mse7 = mdl_7.mse(X_7, Y_7_bits)
    acc_clean, pairs = evaluate_bits4(mdl_7, X_7)
    print(f"7 segmentos — MSE: {mse7:.4f}")
    print(f"Acurácia (sem ruído): {acc_clean*100:.1f}%")
    acc_noise = accuracy_with_noise_bits4(mdl_7, X_7, trials=50, flip_prob=0.10)
    print(f"Acurácia com ruído (p=0.10): {acc_noise*100:.1f}%")
    # Matriz de confusão limpa
    mat = [[0]*10 for _ in range(10)]
    for d, pred in pairs:
        mat[d][pred] += 1
    print("Matriz de confusão (lin=verdade, col=prev):")
    for i in range(10):
        print(mat[i])

    append_txt("relatorio_resultados.txt", f"7-seg — MSE: {mse7:.6f}\n")
    append_txt("relatorio_resultados.txt", f"Acurácia limpa: {acc_clean*100:.1f}%\n")
    append_txt("relatorio_resultados.txt", f"Acurácia com ruído (p=0.10): {acc_noise*100:.1f}%\n")
    append_txt("relatorio_resultados.txt", "Matriz de confusão (limpa):\n")
    for i in range(10):
        append_txt("relatorio_resultados.txt", str(mat[i]) + "\n")

    def run_setup(output_mode: str, hidden_act: str, noise_test_ps=(0.0, 0.05, 0.10, 0.20)) -> Dict:
        if output_mode == "bits4":
            sizes = [7, 5, 4]
            Y = Y_7_bits
            eval_fn = evaluate_bits4
            acc_noise_fn = accuracy_with_noise_bits4
        else:
            sizes = [7, 5, 10]
            Y = Y_7_onehot
            eval_fn = evaluate_onehot
            acc_noise_fn = accuracy_with_noise_onehot
        mdl = MLP(sizes, seed=456, hidden_act=hidden_act, output_act="sigmoid")
        for _ in range(EPOCHS_7SEG):
            train_epoch(mdl, X_7, Y, lr=LR, noise_p=NOISE_TRAIN_P)
        mse = mdl.mse(X_7, Y)
        acc, _ = eval_fn(mdl, X_7)
        accs_noise = {p: acc_noise_fn(mdl, X_7, trials=50, flip_prob=p) for p in noise_test_ps}
        return {"mode": output_mode, "act": hidden_act, "mse": mse, "acc_clean": acc, "acc_noise": accs_noise}

    setups = [
        ("bits4", "sigmoid"),
        ("bits4", "relu"),
        ("onehot10", "sigmoid"),
        ("onehot10", "relu"),
    ]

    print("\n=== COMPARAÇÃO FINAL ===")
    rows = [("saida","oculta","p_ruido","acuracia")]  # para CSVs
    cmp_lines: List[str] = []  # para anexar no TXT

    results_cache = []
    for om, act in setups:
        res = run_setup(om, act)
        results_cache.append(res)
        s1 = f"\nSaída={om}, Oculta={act}"
        s2 = f"MSE: {res['mse']:.6f} | Acurácia limpa: {res['acc_clean']*100:.1f}%"
        s3 = "Acurácia com ruído: " + str({p: f"{v*100:.1f}%" for p, v in res["acc_noise"].items()})
        print(s1); print(s2); print(s3)
        cmp_lines.extend([s1, s2, s3])
        for p, accp in res["acc_noise"].items():
            rows.append((om, act, p, accp))

    # CSV 1 (em %)
    with open("robustez.csv", "w", encoding="utf-8") as f:
        f.write("saida,oculta,p_ruido,acuracia\n")
        for om, act, p, accp in rows[1:]:
            f.write(f"{om},{act},{p:.2f},{accp*100:.1f}%\n")
    print('Arquivo "robustez.csv" salvo.')

    # CSV 2 (fração 0..1)
    with open("robustez_resumo_do_usuario.csv", "w", encoding="utf-8") as f:
        f.write("saida,oculta,p_ruido,acuracia\n")
        for om, act, p, accp in rows[1:]:
            f.write(f"{om},{act},{p:.2f},{accp:.3f}\n")
    print('Arquivo "robustez_resumo_do_usuario.csv" salvo.')

    # Plot
    try:
        from collections import defaultdict
        import matplotlib.pyplot as plt
        series = defaultdict(list)
        for om, act, p, accp in rows[1:]:
            series[(om, act)].append((p, accp))
        plt.figure(figsize=(7,5))
        for (om, act), pts in series.items():
            pts.sort(key=lambda t: t[0])
            xs = [p for p,_ in pts]
            ys = [a for _,a in pts]
            plt.plot(xs, ys, marker="o", label=f"{om} / {act}")
        plt.title("Robustez a ruído no display de 7 segmentos")
        plt.xlabel("Probabilidade de flip de segmento (p)")
        plt.ylabel("Acurácia")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.savefig("robustez_plot.png", bbox_inches="tight")
        plt.close()
        print('Arquivo "robustez_plot.png" salvo.')
    except Exception as e:
        print("Aviso: não foi possível gerar robustez_plot.png:", e)

    # Completar TXT com a comparação final
    try:
        append_txt("relatorio_resultados.txt", "\n\n=== COMPARAÇÃO FINAL ===\n")
        for line in cmp_lines:
            append_txt("relatorio_resultados.txt", line + "\n")
        append_txt("relatorio_resultados.txt", "\nArquivos gerados: robustez.csv, robustez_resumo_do_usuario.csv, robustez_plot.png\n")
        print('Comparação final anexada a "relatorio_resultados.txt".')
    except Exception as e:
        print("Aviso: não consegui anexar ao TXT:", e)
