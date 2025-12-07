# Bayesian CT Reconstruction / ベイズ CT 再構成フレームワーク

本リポジトリは，CT 画像再構成のための計算フレームワークであり，  
古典的な Filtered Backprojection（FBP）手法と，マルコフ確率場(MRF)モデルを統合した  
ベイズ再構成法を使用している．  
C++ によるRadon 変換，パラメータ推定，画像再構成と Python の実験環境を組み合わせ，高解像度画像に対する再構成実験を効率的に実施できる．

---

## 1. 理論背景（ベイズ再構成と自由エネルギー）

本プロジェクトで採用するベイズ CT 再構成法は，観測ノイズモデルと MRF モデルに基づく事前分布を組み合わせ，周波数領域で閉形式の MAP 解 を得る枠組みである．

---

### 1.1 観測モデル（周波数空間）

原画像 $\sigma(x,y)$ にラドン変換を施し観測データ $\tau(s,\theta)$ を得る．  
ノイズを加法ガウスノイズとすると，周波数空間では次の観測モデルになる：

$$ \tilde{\tau}(\tilde{s},\theta) = \tilde{\sigma}(\tilde{s},\theta) + \tilde{n}(\tilde{s},\theta) $$

このとき観測誤差に対応するエネルギー関数は

$$
H_{\mathrm{obs}}
 = 4\pi^2 \gamma \int d\theta \int d\tilde{s}\,
 \left|\tilde{\tau} - \tilde{\sigma}\right|^2
$$

で与えられ，$\gamma$ は観測ノイズの分散の逆数に比例する．

---

### 1.2 事前分布（MRF モデル）

画像の滑らかさと振幅抑制を課すため，事前エネルギーを

$$
H_{\mathrm{pri}}(\sigma) = \beta \!\iint |\nabla\sigma|^2\,dxdy + 4\pi h \!\iint |\sigma|^2\,dxdy
$$

と定義する．  
フーリエ空間では

$$
H_{\mathrm{pri}}
 = 4\pi^2 \!\iint (\beta(\tilde{x}^2+\tilde{y}^2)+h)\,
 |\tilde{\sigma}(\tilde{x},\tilde{y})|^2\,d\tilde{x}d\tilde{y}
$$

となり，極座標 $\tilde{x}=\tilde{s}\cos\theta$, $\tilde{y}=\tilde{s}\sin\theta$ に変換すると

$$
H_{\mathrm{pri}}
 = 4\pi^2 \int d\theta \int d\tilde{s}\,
 (\beta\tilde{s}^2+h)\,|\tilde{s}|\,|\tilde{\sigma}(\tilde{s},\theta)|^2
$$

となる．  
従って事前分布は

$$
p(\sigma\mid\beta,h)
 \propto \exp\!\left[-4\pi^2\!\int\!\!\int
   (\beta\tilde{s}^2+h)\,|\tilde{s}|\,|\tilde{\sigma}|^2
 \right]
$$

で与えられる．

---

### 1.3 MAP 推定とベイズフィルタ（解析解）

観測モデルと事前分布を組み合わせた事後分布は

$$
p(\sigma\mid\tau,\gamma,\beta,h)
\propto
\exp\!\left[
 -4\pi^2\!\int\!\!\int
 \Big(
   \gamma|\tilde{\tau}-\tilde{\sigma}|^2
   +(\beta\tilde{s}^2+h)|\tilde{s}|\,|\tilde{\sigma}|^2
 \Big)
\right]
$$

と書ける．  
ここで

$$
F_{\tilde{s}}
 = (\beta\tilde{s}^2+h)|\tilde{s}|+\gamma
$$

と定義すると，MAP 推定（事後分布最大化）の閉形式解は

$$
\hat{\sigma}(\tilde{s},\theta)
 = \frac{\gamma}{F_{\tilde{s}}}\,\tilde{\tau}(\tilde{s},\theta)
$$

となる．  
この $\gamma/F_{\tilde{s}}$ が **周波数領域のベイズフィルタ**であり，FBP フィルタの一般化に相当する．

---

### 1.4 ハイパーパラメータ推定（自由エネルギー最小化）

ハイパーパラメータ $(\gamma,\beta,h)$ は，周辺尤度

$$
p(\tau\mid\gamma,\beta,h)
 = \int p(\tau\mid\sigma,\gamma)\,p(\sigma\mid\beta,h)\,d\sigma
$$

から定義される **自由エネルギー**

$$
\mathcal{F}(\gamma,\beta,h)
 = -\log\,p(\tau\mid\gamma,\beta,h)
$$

を最小化することで求められる．  

離散化後の自由エネルギーは次の形に整理できる：

$$
F(\gamma,\beta,h) = -\frac{1}{2} \sum_{\tilde{k},l} \left[ \log\left( \frac{8\pi\Delta_\theta\Delta_s}{N_s}\, \gamma\left(1-\frac{\gamma}{F_{\tilde{s}}}\right) \right) - \frac{8\pi^2\Delta_\theta\Delta_s}{N_s}\, \gamma\left(1-\frac{\gamma}{F_{\tilde{s}}}\right) |\tilde{\tau}_{\tilde{k},l}|^2 \right]
$$

この自由エネルギーを最小化することで  
ハイパーパラメータ $(\gamma,\beta,h)$ を推定する．

本リポジトリでは，この自由エネルギーを最小化するために  
**逐次的なグリッドサーチ** を採用しており，  
MAP 推定とハイパーパラメータ推定を一貫した形で自動化している．


---

## 2. リポジトリ構成

```
.
├── bayesian_ct_recon.ipynb      # 実験用メインノートブック
├── recon.cpp                     # C++ 実装（Radon・逆投影）
├── librecon_cpp.dylib            # 共有ライブラリ（macOS）
├── sinograms/                    # サイノグラム生成先
├── results/                      # 再構成結果まとめ
└── README.md
```

---

## 3. 特徴

### 3.1 サイノグラム生成
- C++ による高速 Radon 変換
- ノイズモデル：normal，poisson，delta  
- パラメータ：psd，Ns，Nθ，T（Sparse-view CT）

### 3.2 再構成
**FBP フィルタ**
- Ramp  
- Shepp–Logan  
- Hann  
- Hamming  

**Bayesian Reconstruction**
- MAP 解の閉形式フィルタ  
- 自由エネルギー最小化に基づく (γ, β, h) 推定  
- C++ ＋ Python のハイブリッド実装  

### 3.3 評価
- PSNR，SSIM，RMSE  
- 視覚評価  
- 断面プロファイル比較  

---

## 4. C++ ライブラリのビルド（macOS）

```
clang++ -std=c++17 -O3 -dynamiclib -o librecon_cpp.dylib recon.cpp
```

---

## 5. Python からの利用例

```
import ctypes
lib = ctypes.CDLL("./librecon_cpp.dylib")
```

逆投影の呼び出し例：

```
out = np.zeros((ny,nx), dtype=np.float64)
lib.reconstruction_cpp(
    ny, nx, left, right, top, bottom,
    sino.ctypes.data_as(...),
    Nθ, Ns, ds,
    out.ctypes.data_as(...)
)
```

---

## 6. 数値実験パイプライン

Notebook 内では次のように数値実験を一括実行できる．

```
records = numerical_experiment(
    psd_range=(0.5,4.0,0.5),
    N_sizes=256,
    N_theta=256,
    T_range=(1.0,10.5,1.0),
    filters=("bayes","ramp","shepp-logan","hann","hamming"),
    phantom=phantom,
    radon_transform_fn=radon_transform_cxx,
    free_energy_fn=free_energy,
    n_section_search_fn=n_section_search_cxx,
    image_reconstruction_fn=image_reconstruction,
)
```

### 実行時間の目安
- 256×256，180 投影：**1 枚あたり約 0.7 秒**  
- 2048×2048，1800 投影：**1 枚あたり約 16 秒**

---

## 7. 必要環境

```
python >= 3.9
numpy
scipy
matplotlib
pandas
tqdm
scikit-image
```

---

## 8. 参考文献

1. A. C. Kak and M. Slaney,  
   *Principles of Computerized Tomographic Imaging*,  
   IEEE Press, 1988.

2. G. N. Ramachandran and A. V. Lakshminarayanan,  
   “Three-dimensional reconstruction from radiographs and electron micrographs: Application of convolutions instead of Fourier transforms,”  
   *Proceedings of the National Academy of Sciences of the USA*,  
   vol. 68, no. 9, pp. 2236–2240, 1971.

3. H. Shouno and M. Okada,  
   “Bayesian Image Restoration for Medical Images Using Radon Transform,”  
   *Journal of the Physical Society of Japan*,  
   vol. 79, no. 7, p. 074004, 2010.

---