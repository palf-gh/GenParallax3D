# GenParallax3D

[日本語](#ja) | [English](#en)

<a id="ja"></a>

## 日本語

レイヤー構造を活かした3Dパララックス再構成ツールです。PSD/PSBを読み込み、After Effectsでのカメラワーク用にGLB（glTF 2.0）を書き出します。

### ダウンロード（Pythonは不要）

[Releases](https://github.com/palf-gh/GenParallax3D/releases) から、使っているOS用のzipをダウンロードします。

- **Windows：** `GenParallax3D-<版>-windows-x64.zip` を展開し、フォルダ内の `GenParallax3D.exe` を実行します。フォルダごと置いておいてください（exeだけを移すと動きません）。署名していないため、初回は SmartScreen が出ることがあります。「詳細情報」→「実行」で起動できます。
- **macOS：** Apple Silicon は `…-macos-arm64.zip`、Intel は `…-macos-x64.zip` です。展開した `GenParallax3D.app` を「アプリケーション」に移します。署名・公証していないため、初回は Finder で右クリック →「開く」で起動します（macOS 15 以降は、システム設定 ▸ プライバシーとセキュリティ の「このまま開く」）。
- 単体版には AI 深度推定（Depth Anything V2、torch + transformers）は入っていません。深度マップの画像を「深度 ▸ 深度マップを読み込む…」で読み込んで使います。AI推定が必要な場合は、下の方法でソースから実行してください。
- 動作確認：`GenParallax3D --self-test report.txt` で、同梱ライブラリとOpenGLを確認し結果をファイルに書きます（終了コード0で成功）。

単体版は `python packaging/build.py` で作れます（AI機能を入れていない環境で実行してください）。タグ `v*` をプッシュすると GitHub Actions が Windows と macOS 版を作り、リリースに添付します。

### 実行方法

```bash
pip install -e .[dev]          # Depth Anything V2 を使う場合は [ai] も追加（torch + transformers）
python -m genparallax [file.psd | project.gp3d]
python -m pytest
```

### 構成

| モジュール | 役割（計画書 §2） |
|---|---|
| `genparallax/scene.py` | 共通のシーンモデル（Scene、Camera、ReconstructionUnit、Mesh）と座標系の約束事 |
| `genparallax/document/psd_loader.py` | ドキュメント層：psd-tools による解析、階層、クリッピング、グループの不透明度とマスク、互換性の警告 |
| `genparallax/document/psd_compat.py` | psd-tools の互換対処と高速化（Pillow 12 対応、「ZIP（予測付き）」の展開をNumPy化） |
| `genparallax/reconstruction/depth.py` | 深度マップの読み込み、近さ→Zの変換、隠れ領域を除いた凹凸の算出、穴埋め |
| `genparallax/reconstruction/units.py` | 再構成単位の作成・統合・分割、可視領域、奥行きの割り当て、オーバーレイの判定 |
| `genparallax/reconstruction/constraints.py` | レイヤー順の拘束（局所的な押し込み）、オーバーレイの配置、裏スレッド用のスナップショット |
| `genparallax/reconstruction/meshing.py` | 均一メッシュと適応メッシュ（2:1でバランスした四分木）、逆投影、CPUでの深度ラスタライズ |
| `genparallax/reconstruction/estimator.py` | Depth Anything V2 Small（別プロセスで実行、結果をキャッシュ） |
| `genparallax/rendering/` | ModernGL レンダラー（Qtに依存せず、テストでもヘッドレスで使用）と、オービットカメラ |
| `genparallax/export/` | `glb.py`：pygltflib による書き出し（Unlit、不透明/半透明の分割、基準カメラ）。`ae.py`：After Effects 用JSON |
| `genparallax/project.py` | `.gp3d` プロジェクトファイル（形式v2）とPSDの再読み込み。AI深度は `.gp3d_cache/` にキャッシュ |
| `genparallax/app/` | PySide6 GUI（レイヤーツリー、QOpenGLWidget のビューポート、プロパティ、日本語/英語の切り替え） |

### 約束事

- カメラ座標は glTF に合わせています。+Y が上で、基準カメラは原点から −Z 方向を向きます。
- 各単位の深度は `Z = B + S·R(u, v) + P(u, v)` です。各頂点は元の画素を通る基準カメラの視線上に置くので、基準ビューは常にPSDを再現します。
  - `R`（凹凸）は深度マップから求めます。使うのは、その単位が統合画像で実際に見えている画素だけです。
  - `P` は、レイヤーが重なる場所でPSDの重なり順を保つための押し込み量です。設定から導出されるため保存しません。
- 深度マップの「近さ」は `1/Z` について線形に対応させます。近さ1が `Z = f`、近さ0が `Z = f × シーンの奥行き範囲` です。
- 基準面（`Z = f`）では1単位が1画素に当たり、`f` は After Effects のカメラのズーム値と同じです。書き出すGLBは、After Effects のモデル設定（「書き出しのモデル単位」と同じモデル単位、オブジェクトスケール100%）で読み込むとコンポの1 pxがAEの1 pxになる大きさで書き出します（AEは1 m＝512 pxとして読み込むため、初期値のメートルでは1 px＝1/512 m）。初期設定では基準面が z = 0 に来るよう平行移動します。
- **オーバーレイ** は、光芒・靄・柔らかい花など、主に半透明のレイヤーです。
  - 自動で判定し、単位ごとに切り替えられます。
  - 深度マップではなく、下にあるレイヤーから奥行きを決めます。
  - 下のレイヤーを押し込むことはありません。
  - 深度マップを読み込む前は、下のレイヤーに実の凹凸がないため、オーバーレイも平面のままにします。

### このPCでの Depth Anything

transformers は、torchvision がインストールされていると必ず読み込みます。torch と合わないビルドの torchvision は `operator torchvision::nms does not exist` で失敗します。そこで推定処理は次のようにしています。
- 前処理を自前で行う
- 壊れた torchvision を、推定用のワーカープロセスの中だけで隠す

torch と torchvision を揃えたビルド（GPU推論にはCUDA版）を入れれば、この回避策は不要になります。

### 現状

- **フェーズ1〜2：** 実装・テスト済みです。
- **フェーズ3：** Depth Anything V2 Small を実PSDで確認しました（CPUでモデルのダウンロード込み約1分）。
- **フェーズ4：** 実装済みで、11レイヤーの実PSDで確認しました。アプリ内の基準ビューとPSD統合画像の差は平均0.6/255です。対応している機能は次のとおりです。
  - 隠れ領域を除いた奥行きの割り当て
  - レイヤー順の拘束
  - オーバーレイ
  - 輪郭の内側オフセット
  - 適応メッシュ
  - 深度差分表示
  - PSDの再読み込みと自動再読み込み
- **外部のglTFレンダラー：** レンダラーが自分で物体を並べ替えると、半透明レイヤーが背後のものを隠すことがあります。
  - 初期設定では、各単位を2つのプリミティブで書き出します。不透明な部分はアルファテスト（MASK）、半透明の縁はブレンドです。
  - オーバーレイはすべてブレンドのままです。
  - ファイル ▸ 書き出すGLBの透明表現 で切り替えられます。
  - three.js の標準の並べ替えのままで、実PSDとの差は平均0.73/255（16/255を超える画素は0.2%）です。すべてブレンドにすると10.6/255です。
  - After Effects での確認はまだです。
- **操作性**（11レイヤー、1494×840のPSDでの値）：
  - PSDを開く、深度の適用、再読み込み、書き出しは裏のスレッドで動きます。その間は編集だけを止め、ウィンドウとビューポートは反応し続けます（停止は0.3秒未満）。
  - スライダーの編集は約1msでメッシュに反映されます。
  - レイヤー順の処理は裏のスレッドで、最短300msおきに実行します。スライダーをドラッグしている間も止まりません。
  - 実行中に編集されなかった単位には結果をすぐに反映するので、ドラッグに合わせて下のレイヤーが追従します。
- **UIの言語：** 初期設定は日本語で、表示 ▸ 言語 で切り替えられます（再起動が必要）。コード内の文字列はイギリス英語で、`genparallax/app/i18n.py` で翻訳しています。
- **フェーズ5（After Effects での確認）：** まだ始めていません。

### 性能に関するメモ

- `document/psd_compat.py` は、「ZIP（予測付き）」チャンネルを展開する psd-tools の純Python実装を、NumPy版に置き換えます。出力はバイト単位で同一で、大きなファイルの読み込みは約4倍速くなります。
- 各メッシュの「画素→三角形」の対応表は、メッシュの位相ごとに1回だけ計算します。深度を編集したときは `1/Z` を補間し直すだけです。
- レイヤー順の処理は、自分の設定と、上に重なるレイヤーのメッシュ版番号が変わった単位だけをやり直します。

---

<a id="en"></a>

## English

Layer-aware 3D parallax reconstruction: PSD/PSB in, GLB (glTF 2.0) out, for camera moves in After Effects.

### Download (no Python needed)

Get the zip for your system from [Releases](https://github.com/palf-gh/GenParallax3D/releases).

- **Windows:** unzip `GenParallax3D-<version>-windows-x64.zip` and run `GenParallax3D.exe` inside the folder. Keep the folder together; the exe does not work on its own. The app is unsigned, so SmartScreen may appear the first time: choose "More info" → "Run anyway".
- **macOS:** `…-macos-arm64.zip` for Apple Silicon, `…-macos-x64.zip` for Intel. Move `GenParallax3D.app` to Applications. It is neither signed nor notarised, so open it the first time with right-click → Open (on macOS 15 and later: System Settings ▸ Privacy & Security ▸ "Open Anyway").
- The standalone app does not include AI depth estimation (Depth Anything V2, torch + transformers). Load a depth map image with Depth ▸ Load Depth Map… instead, or run from source with the `[ai]` extra.
- Health check: `GenParallax3D --self-test report.txt` exercises the bundled libraries and OpenGL and writes the result to the file (exit code 0 means it passed).

Build it yourself with `python packaging/build.py` (in an environment without the AI extra). Pushing a `v*` tag makes GitHub Actions build the Windows and macOS apps and attach them to a release.

### Run

```bash
pip install -e .[dev]          # add [ai] for Depth Anything V2 (torch + transformers)
python -m genparallax [file.psd | project.gp3d]
python -m pytest
```

### Layout

| Module | Role (plan §2) |
|---|---|
| `genparallax/scene.py` | Shared scene model: Scene, Camera, ReconstructionUnit, Mesh, and the coordinate conventions |
| `genparallax/document/psd_loader.py` | Document layer: psd-tools parsing, hierarchy, clipping, group opacity/masks, compatibility warnings |
| `genparallax/document/psd_compat.py` | psd-tools compatibility and speed patches (Pillow 12, NumPy "ZIP with prediction" decoder) |
| `genparallax/reconstruction/depth.py` | Depth-map loading, nearness → Z, occlusion-aware relief, inpainting |
| `genparallax/reconstruction/units.py` | Units: creation, merge/split, visibility, depth assignment, overlay detection |
| `genparallax/reconstruction/constraints.py` | Layer-order constraints (local push-back), overlay placement, snapshots for background passes |
| `genparallax/reconstruction/meshing.py` | Uniform and adaptive (2:1-balanced quadtree) meshing, back-projection, CPU depth rasteriser |
| `genparallax/reconstruction/estimator.py` | Depth Anything V2 Small (worker process, cached) |
| `genparallax/rendering/` | ModernGL renderer (Qt-independent, also used headlessly by tests) and the orbit camera |
| `genparallax/export/` | `glb.py`: pygltflib writer (unlit, opaque/translucent split, reference camera). `ae.py`: After Effects JSON |
| `genparallax/project.py` | `.gp3d` project files (format v2), PSD reload; AI depth cached in `.gp3d_cache/` |
| `genparallax/app/` | PySide6 GUI: layer tree, QOpenGLWidget viewport, properties, Japanese/English UI |

### Conventions

- Camera space follows glTF: +Y up, the reference camera at the origin looking down −Z.
- Depth per unit: `Z = B + S·R(u, v) + P(u, v)`. Each vertex sits on the reference ray through its source pixel, so the reference view always reproduces the PSD.
  - `R` is measured from the depth map, using only the pixels where the unit is visible in the composite.
  - `P` is the push-back field that keeps the PSD stacking order where layers overlap. It is derived data and is not saved.
- Depth-map nearness maps linearly in `1/Z`: nearness 1 → `Z = f`, nearness 0 → `Z = f × Scene Depth Range`.
- One unit equals one pixel at the reference plane (`Z = f`), and `f` is After Effects' camera Zoom. The exported GLB is scaled so that After Effects, with the same Model Units as *Export Model Units (AE)* and Object Scale 100 %, imports one composition pixel as one AE pixel (AE reads 1 m as 512 px, so the default, metres, is 1/512 m per px). By default it is shifted so the reference plane lands at z = 0.
- **Overlays** are mostly translucent layers: light rays, haze, soft flowers. They are detected automatically and can be toggled per unit. An overlay takes its depth from the layers below it instead of from the depth map, and never pushes them back. Until a depth map is loaded the layers below have no relief, so the overlay stays flat.

### Depth Anything on this machine

transformers imports torchvision whenever it is installed. A torchvision build that does not match torch fails with `operator torchvision::nms does not exist`. The estimator therefore:
- does its own preprocessing,
- hides a broken torchvision inside its worker process.

Installing matching `torch`/`torchvision` builds (with CUDA, for GPU inference) removes the need for this workaround.

### Status

- **Phases 1–2:** implemented and tested.
- **Phase 3:** Depth Anything V2 Small has been verified on a real PSD (CPU: about 1 minute including the model download).
- **Phase 4:** implemented and checked on a real 11-layer PSD. On that PSD the in-app reference render differs from the PSD composite by a mean of 0.6/255. It covers:
  - occlusion-aware depth assignment
  - layer-order constraints
  - overlays
  - edge inset
  - adaptive meshing
  - Depth Difference view
  - PSD reload and auto-reload
- **External glTF renderers:** translucent layers can hide what lies behind them when a renderer sorts objects itself.
  - By default each unit is exported as two primitives: an alpha-tested (MASK) part for opaque texels, and a blended part for the translucent edges.
  - Overlays stay fully blended.
  - Toggle with File ▸ Transparency in Exported GLB.
  - With three.js's default sorting the real PSD reproduces at 0.73/255 mean (0.2 % of pixels > 16/255). Blending everything gives 10.6/255.
  - After Effects itself is still unverified.
- **Interactivity** (11-layer 1494×840 PSD):
  - Opening a PSD, applying depth, reloading and exporting run in a background thread. Editing is locked meanwhile, but the window and the viewport keep responding (stalls < 0.3 s).
  - Slider edits update the mesh in about 1 ms.
  - The layer-order pass runs in a worker thread at most every 300 ms, also while a slider is being dragged.
  - Results for units not edited meanwhile are applied at once, so the layers below follow a drag.
- **UI language:** Japanese by default; switch with View ▸ Language (restart needed). Source strings are British English, translated in `genparallax/app/i18n.py`.
- **Phase 5 (After Effects verification):** not started.

### Performance notes

- `document/psd_compat.py` replaces psd-tools' pure-Python decoder for "ZIP with prediction" channels with NumPy. The output is byte-identical, and loading is about 4× faster on large files.
- The pixel → triangle coverage of each mesh is computed once per topology. Depth edits then only re-interpolate `1/Z`.
- The layer-order pass only redoes units whose own settings, or whose overlapping upper layers' mesh revisions, changed.
