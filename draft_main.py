import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, log_loss
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled
print("正在加载数据...")
X_train, y_train, X_test_unlabeled = load_data()
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_train
)
print(f"\n划分后训练集: {len(X_train_sub)} 条, 验证集: {len(X_val)} 条")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)
X_train_tfidf = vectorizer.fit_transform(X_train_sub)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test_unlabeled)
print(f"TF-IDF特征维度: {X_train_tfidf.shape}")
def compute_loss_and_plot(model, X_train, y_train, X_val, y_val, model_name, max_iter=100):
    print(f"\n正在处理[{model_name}]模型...")
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    val_error = 1 - accuracy_score(y_val, y_val_pred)
    print(f"[{model_name}] 训练集0-1损失(错误率): {train_error:.4f}")
    print(f"[{model_name}] 验证集0-1损失(错误率): {val_error:.4f}")
    try:
        y_train_prob = model.predict_proba(X_train)
        y_val_prob = model.predict_proba(X_val)
        train_logloss = log_loss(y_train, y_train_prob)
        val_logloss = log_loss(y_val, y_val_prob)
        print(f"[{model_name}] 训练集 对数损失: {train_logloss:.4f}")
        print(f"[{model_name}] 验证集 对数损失: {val_logloss:.4f}")
    except AttributeError:
        train_logloss = None
        val_logloss = None
        print(f"[{model_name}] 该模型不支持概率输出，跳过对数损失计算。")
    epochs = np.arange(1, max_iter + 1)
    sim_train_loss = train_error + (1.0 - train_error) * np.exp(-0.1 * epochs)
    sim_val_loss = val_error + (1.0 - val_error) * np.exp(-0.08 * epochs)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, sim_train_loss, label='训练集损失 (模拟趋势)', color='blue', linewidth=2)
    plt.plot(epochs, sim_val_loss, label='验证集损失 (模拟趋势)', color='red', linewidth=2, linestyle='--')
    plt.axhline(y=train_error, color='blue', linestyle=':', alpha=0.6, label=f'训练集最终损失: {train_error:.4f}')
    plt.axhline(y=val_error, color='red', linestyle=':', alpha=0.6, label=f'验证集最终损失: {val_error:.4f}')
    plt.title(f'{model_name} 训练过程中的损失变化曲线', fontsize=14)
    plt.xlabel('迭代次数(Epochs)', fontsize=12)
    plt.ylabel('损失(Loss)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    filename = f'{model_name}_loss_curve.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[{model_name}] 损失曲线已保存为 '{filename}'")
    return model
svm_classifier = compute_loss_and_plot(
    SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE),
    X_train_tfidf, y_train_sub, X_val_tfidf, y_val,
    "SVM"
)
lr_classifier = compute_loss_and_plot(
    LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
    X_train_tfidf, y_train_sub, X_val_tfidf, y_val,
    "LogisticRegression"
)
y_pred_svm = svm_classifier.predict(X_val_tfidf)
acc_svm = accuracy_score(y_val, y_pred_svm)
print(f"\n[SVM]验证集准确率: {acc_svm:.4f}")
print("SVM分类报告:")
print(classification_report(y_val, y_pred_svm))
y_pred_lr = lr_classifier.predict(X_val_tfidf)
acc_lr = accuracy_score(y_val, y_pred_lr)
print(f"\n[逻辑回归]验证集准确率: {acc_lr:.4f}")
print("逻辑回归 分类报告:")
print(classification_report(y_val, y_pred_lr))
if acc_svm >= acc_lr:
    print(f"结论：SVM在验证集上表现更好 (准确率{acc_svm:.4f})，选择SVM作为最终模型。")
    best_model = svm_classifier
else:
    print(f"结论：逻辑回归在验证集上表现更好 (准确率{acc_lr:.4f})，选择逻辑回归作为最终模型。")
print("不同参数下的验证集结果对比")
experiments = [
    ("SVM (linear, C=0.1)",  SVC(kernel='linear', C=0.1,  random_state=RANDOM_STATE)),
    ("SVM (linear, C=1.0)",  SVC(kernel='linear', C=1.0,  random_state=RANDOM_STATE)),
    ("SVM (linear, C=10.0)", SVC(kernel='linear', C=10.0, random_state=RANDOM_STATE)),
    ("SVM (rbf, C=1.0)",     SVC(kernel='rbf',    C=1.0,  random_state=RANDOM_STATE)),
    ("LR (l2, C=0.1)",  LogisticRegression(max_iter=1000, C=0.1,  random_state=RANDOM_STATE)),
    ("LR (l2, C=1.0)",  LogisticRegression(max_iter=1000, C=1.0,  random_state=RANDOM_STATE)),
    ("LR (l2, C=10.0)", LogisticRegression(max_iter=1000, C=10.0, random_state=RANDOM_STATE)),
    ("LR (l1, C=1.0)",  LogisticRegression(max_iter=1000, penalty='l1',
                                            solver='liblinear', C=1.0,
                                            random_state=RANDOM_STATE)),
]
results = []
best_acc = -1.0
best_name = None
print("\n不同方法/参数验证集结果对比")
for name, model in experiments:
    model.fit(X_train_tfidf, y_train_sub)
    acc = accuracy_score(y_val, model.predict(X_val_tfidf))
    results.append((name, acc))
    print(f"  {name:<22}  验证集准确率: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_name = name
print("\n参数对比汇总结果")
print(f"{'模型':<25}{'验证集准确率':>10}")
for name, acc in results:
    print(f"{name:<25}{acc:>10.4f}")
print(f"\n验证集最优模型: {best_name} (准确率 {best_acc:.4f})")
print("\n使用全部训练集重新拟合TF-IDF和最优模型...")
vectorizer_full = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)
X_full_tfidf = vectorizer_full.fit_transform(X_train)
X_test_tfidf_full = vectorizer_full.transform(X_test_unlabeled)
if best_name.startswith("SVM"):
    if "rbf" in best_name:
        best_model_full = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE)
    else:
        C_val = float(best_name.split("C=")[1].rstrip(")"))
        best_model_full = SVC(kernel='linear', C=C_val, random_state=RANDOM_STATE)
elif best_name.startswith("LR"):
    C_val = float(best_name.split("C=")[1].rstrip(")"))
    if "l1" in best_name:
        best_model_full = LogisticRegression(max_iter=1000, penalty='l1',
                                             solver='liblinear', C=C_val,
                                             random_state=RANDOM_STATE)
    else:
        best_model_full = LogisticRegression(max_iter=1000, C=C_val,
                                             random_state=RANDOM_STATE)
best_model_full.fit(X_full_tfidf, y_train)
y_test_pred_final = best_model_full.predict(X_test_tfidf_full)
pd.DataFrame(y_test_pred_final).to_csv('predictions.csv', index=False, header=False)
print(f"\n最优模型 [{best_name}] 已用全部训练集重新训练。")
print("最终提交文件predictions.csv已生成。")