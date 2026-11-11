"""User-interface translation.

Source strings in the code are British English. ``tr()`` returns the string in the
current language (Japanese by default) and fills ``{placeholders}``. Messages that
come back from the library with variable parts (warnings, conflict reports) are
translated by ``tr_message()``, which matches them against the same templates.
"""

from __future__ import annotations

import re

LANGUAGES = {"ja": "日本語", "en": "English"}
_language = "ja"


def set_language(code: str) -> None:
    global _language
    _language = code if code in LANGUAGES else "ja"


def language() -> str:
    return _language


def tr(text: str, **values) -> str:
    """Translate ``text`` (an English template) and format it with ``values``."""
    template = JA.get(text, text) if _language == "ja" else text
    return template.format(**values) if values else template


def tr_message(message: str) -> str:
    """Translate a message produced by the library (variable parts kept as they are)."""
    if _language != "ja":
        return message
    if message in JA:
        return JA[message]
    for pattern, template in _patterns():
        m = pattern.fullmatch(message)
        if m:
            return template.format(**{k: tr_message(v) if k == "message" else v for k, v in m.groupdict().items()})
    return message


_compiled: list[tuple[re.Pattern, str]] | None = None


def _patterns() -> list[tuple[re.Pattern, str]]:
    global _compiled
    if _compiled is None:
        _compiled = []
        for english in LIBRARY_MESSAGES:
            regex = re.escape(english)
            regex = re.sub(r"\\\{(\w+)\\\}", lambda m: f"(?P<{m.group(1)}>.+?)", regex)
            _compiled.append((re.compile(regex, re.DOTALL), JA[english]))
    return _compiled


#: Library messages with variable parts (translated by pattern).
LIBRARY_MESSAGES = [
    # psd_loader
    "Full composite failed ({error}); using embedded preview.",
    "Adjustment layer: cannot be reconstructed in 3D (excluded).",
    "Blend mode '{mode}' is rendered as Normal in 3D.",
    "Group blend mode '{mode}' is ignored; children are reconstructed individually.",
    "Clipping layer without a base layer.",
    "Clipped to a group: not supported yet, the layer is ignored.",
    "Translucent group: opacity is applied per layer, so overlapping children look lighter than in Photoshop.",
    # project
    "PSD size changed from {old} to {new}.",
    "Cached depth map is missing; run depth estimation again.",
    "The depth map does not match the PSD size and was discarded.",
    "Layer '{path}' no longer exists in the PSD.",
    "New layer '{path}' added.",
    "'{name}' was moved back {px} px to keep the PSD layer order.",
    "The image changed; the depth map may be out of date (re-run depth estimation if needed).",
    # "<layer path>: <message>" prefix used by the PSD warnings (tried last)
    "{path}: {message}",
]

JA: dict[str, str] = {
    # ---- library messages ------------------------------------------------
    "Full composite failed ({error}); using embedded preview.": "統合画像の合成に失敗したため（{error}）、埋め込みプレビューを使います。",
    "Adjustment layer: cannot be reconstructed in 3D (excluded).": "調整レイヤーは3D化できないため除外しました。",
    "Blend mode '{mode}' is rendered as Normal in 3D.": "描画モード「{mode}」は3Dでは「通常」として描画されます。",
    "Group blend mode '{mode}' is ignored; children are reconstructed individually.":
        "グループの描画モード「{mode}」は無視し、中のレイヤーを個別に再構成します。",
    "Clipping layer without a base layer.": "クリッピング先のないクリッピングレイヤーです。",
    "Clipped to a group: not supported yet, the layer is ignored.":
        "グループへのクリッピングは未対応のため、このレイヤーは無視します。",
    "Translucent group: opacity is applied per layer, so overlapping children look lighter than in Photoshop.":
        "半透明グループ：不透明度をレイヤーごとに適用するため、重なった部分がPhotoshopより薄く見えます。",
    "PSD size changed from {old} to {new}.": "PSDのサイズが {old} から {new} に変わりました。",
    "Cached depth map is missing; run depth estimation again.": "保存済みの深度マップが見つかりません。深度推定をやり直してください。",
    "The depth map does not match the PSD size and was discarded.": "深度マップがPSDのサイズと合わないため破棄しました。",
    "Layer '{path}' no longer exists in the PSD.": "レイヤー「{path}」はPSDにもう存在しません。",
    "New layer '{path}' added.": "新しいレイヤー「{path}」を追加しました。",
    "'{name}' was moved back {px} px to keep the PSD layer order.":
        "PSDの重なり順を保つため「{name}」を {px} px 奥へ移動しました。",
    "The image changed; the depth map may be out of date (re-run depth estimation if needed).":
        "画像が変更されました。深度マップが古い可能性があります（必要なら深度推定をやり直してください）。",
    "{path}: {message}": "{path}：{message}",

    # ---- menus -------------------------------------------------------------
    "&File": "ファイル(&F)",
    "&Depth": "深度(&D)",
    "&View": "表示(&V)",
    "&Help": "ヘルプ(&H)",
    "Language": "言語",
    "Open PSD/PSB…": "PSD/PSBを開く…",
    "Open Project…": "プロジェクトを開く…",
    "Reload PSD": "PSDを再読み込み",
    "Reload PSD Automatically When Saved": "PSDが保存されたら自動で再読み込み",
    "Save Project": "プロジェクトを保存",
    "Save Project As…": "名前を付けてプロジェクトを保存…",
    "Export GLB (Combined Scene)…": "GLBを書き出し（1ファイルにまとめる）…",
    "Export GLB (Separate Objects)…": "GLBを書き出し（単位ごとに分ける）…",
    "Transparency in Exported GLB": "書き出すGLBの透明表現",
    "Split Opaque / Translucent (recommended)": "不透明と半透明を分ける（推奨）",
    "Blend Everything": "すべてブレンド",
    "Quit": "終了",
    "Load Depth Map…": "深度マップを読み込む…",
    "Estimate Depth (Depth Anything V2 Small)": "深度を推定（Depth Anything V2 Small）",
    "Clear Depth Map": "深度マップを消去",
    "Assign Depths from Depth Map": "深度マップから奥行きを割り当て直す",
    "Resolve Layer Order": "レイヤー順を整える",
    "Keep Layer Order Automatically": "レイヤー順を自動で保つ",
    "After each edit, push lower layers back where they would show in front of the layers above them in the PSD":
        "編集のたびに、PSDで上にあるレイヤーより手前に出てしまう下のレイヤーを奥へ押し込みます",
    "Reset Default Layer Depths": "レイヤーの奥行きを初期値に戻す",
    "Reset Preview Camera": "プレビューカメラをリセット",
    "Viewport Controls": "ビューポートの操作方法",
    "About": "このアプリについて",
    "Status / Log": "状況 / ログ",
    "Main": "メイン",
    "Open PSD": "PSDを開く",
    "Load Depth": "深度を読み込む",
    "Estimate Depth": "深度を推定",
    "Export GLB": "GLBを書き出す",
    " Display: ": " 表示：",
    "Language changes take effect after restarting the application.": "言語の変更はアプリの再起動後に反映されます。",

    # ---- display modes -------------------------------------------------------
    "Original": "元画像",
    "Reference": "基準カメラ",
    "Free Camera": "自由カメラ",
    "Depth": "深度",
    "Difference": "差分",
    "Depth Difference": "深度差分",
    "Wireframe": "ワイヤーフレーム",
    "Parallax": "パララックス",

    # ---- reconstruction modes --------------------------------------------
    "Flat": "平面",
    "Depth Surface": "深度サーフェス",
    "Smoothed Surface": "なめらかなサーフェス",
    "Manual Depth": "手動深度",
    "Excluded": "除外",

    # ---- layer panel -------------------------------------------------------
    "Layer": "レイヤー",
    "Unit": "再構成単位",
    "Merge into Unit": "1つの単位に統合",
    "Combine the selected layers into a single reconstruction unit": "選択したレイヤーを1つの再構成単位にまとめます",
    "Split Unit": "単位を分割",
    "Split the selected unit back into one unit per layer": "選択した単位をレイヤーごとの単位に戻します",
    "{kind} — {blend}, opacity {opacity}": "{kind} — {blend}、不透明度 {opacity}",
    "bbox {bbox}": "範囲 {bbox}",
    "Hidden in the PSD": "PSDで非表示",

    # ---- properties panel --------------------------------------------------
    "Reconstruction Unit": "再構成単位",
    "Name": "名前",
    "No unit selected": "単位が選択されていません",
    "Reconstruction Mode": "再構成モード",
    "Depth Settings": "深度設定",
    "Base Depth (px)": "基準奥行き (px)",
    "Depth Strength (px)": "凹凸の強さ (px)",
    "Depth Offset (px)": "奥行きオフセット (px)",
    "Depth Smoothing (σ px)": "深度の平滑化 (σ px)",
    "Gaussian σ in image pixels; large values smooth thin lattice layers such as towers":
        "深度マップをぼかすガウスσ（画像px）。鉄塔のような細い格子状のレイヤーでは大きな値が有効です",
    "Lay onto layers below (overlay)": "下のレイヤーに沿わせる（オーバーレイ）",
    "For light rays, haze, scattered petals…: take depth from the layers underneath instead of the depth map, "
    "which only shows what is behind a translucent layer":
        "光芒・靄・散った花びらなど向け：深度マップ（半透明レイヤーの場合は背後の深度しか写らない）ではなく、"
        "下にあるレイヤーから奥行きを決めます",
    "Edge Inset": "輪郭の内側オフセット",
    "Depth near the silhouette is taken from this far inside, so blurred AI depth at the outline does not curl the edge":
        "輪郭付近の深度をこの距離だけ内側から取り、AI深度の輪郭のにじみで縁が反り返るのを防ぎます",
    "Occlusion Inset": "重なり境界の余白",
    "Depth this close to an opaque layer above is not trusted and is filled in from further away, "
    "so the layer does not grow spikes towards that layer":
        "上に重なる不透明レイヤーからこの距離以内の深度は信用せず、離れた所の値で埋めます。"
        "上のレイヤーへ向かって「とげ」が伸びるのを防ぎます",
    "Mesh Density": "メッシュ密度",
    "Grid step; with an adaptive mesh this is the smallest cell": "格子の間隔。適応メッシュでは最小セルの大きさです",
    "Adaptive": "適応",
    "Large cells where the surface is flat, fine cells at silhouettes and depth changes":
        "平らな所は大きなセル、輪郭や深度の変化がある所は細かいセルにします",
    "Mesh Detail": "メッシュ詳細度",
    "Mesh Simplification": "メッシュの簡略化",
    "Off": "オフ",
    "Standard": "標準",
    "Strong": "強",
    "Fewer vertices: large triangles where the surface is flat or smooth. "
    "The outline and the camera view stay the same; Strong allows more depth error":
        "頂点を減らします。平らな所やなめらかな所を大きな三角形にまとめます。"
        "輪郭とカメラからの見た目は変わりません。「強」は奥行きの誤差を大きめに許します",
    "Adaptive tolerance in relief units: smaller = more triangles": "適応メッシュの許容誤差（凹凸の単位）。小さいほど三角形が増えます",
    "Custom Depth Map": "個別の深度マップ",
    "Z Position (px)": "Z位置 (px)",
    "After Effects z: 0 is the composition plane, positive is away from the camera":
        "After Effectsのz座標。0がコンポジション平面、正がカメラから奥です",
    "Camera Distance (px)": "カメラからの距離 (px)",
    "Distance from the camera in composition pixels (Z position + Zoom)":
        "コンポジションpxでのカメラからの距離（Z位置＋ズーム）",
    "Composition": "コンポジション",
    "Composition Size (px)": "コンポジションサイズ (px)",
    "Image Scale (%)": "画像のスケール (%)",
    "Composition pixels per image pixel; the image stays centred":
        "画像1 px当たりのコンポジションのpx数。画像は中央に配置されます",
    "Fit": "フィット",
    "Scale the image so that all of it fits in the composition": "画像全体がコンポジションに収まるようにスケールします",
    "Fill": "フィル",
    "Scale the image so that it covers the composition": "画像がコンポジションを埋めるようにスケールします",
    "100%": "100%",
    "Lens Preset": "レンズのプリセット",
    "After Effects focal-length presets for a 36 mm film": "36 mmフィルムでのAfter Effectsの焦点距離プリセット",
    "Custom": "カスタム",
    "Focal Length (mm)": "焦点距離 (mm)",
    "Angle of View (°)": "画角 (°)",
    "Editing Zoom or Angle of View changes the focal length, as in After Effects":
        "ズームや画角を編集すると、After Effectsと同じく焦点距離が変わります",
    "Zoom": "ズーム",
    "Film Size (mm)": "フィルムサイズ (mm)",
    "Film Size Measured": "フィルムサイズの測定",
    "Horizontal": "水平",
    "Vertical": "垂直",
    "Diagonal": "対角",
    "Units": "単位",
    "px": "px",
    "mm": "mm",
    "Unit used for Zoom and the composition size readout; layer Z stays in pixels":
        "ズームとコンポジションサイズの表示に使う単位。レイヤーのZ位置は常にpxです",
    "Measured Composition Size": "コンポジションサイズ（測定辺）",
    "Size of the composition along the measured film dimension": "フィルムサイズの測定方向に沿ったコンポジションの長さ",
    "Camera Settings": "カメラ設定",
    "Scene Depth Range": "シーンの奥行き範囲",
    "Farthest depth-map value lands at this multiple of the nearest (Zoom) distance.\n"
    "Changing it re-assigns all depths from the depth map.":
        "深度マップで最も遠い値を、最も近い距離（ズーム）の何倍に置くかを指定します。\n"
        "変更すると、すべての奥行きを深度マップから割り当て直します。",
    "Vertical FOV": "垂直画角",
    "Zoom / Focal": "ズーム / 焦点距離",
    "Parallax Amplitude (× width)": "パララックスの振れ幅（×幅）",
    "Parallax Period": "パララックスの周期",
    "Export Model Units (AE)": "書き出しのモデル単位 (AE)",
    "Pick the same Model Units in After Effects' Model Settings and keep Object Scale at 100 %: one composition "
    "pixel is then one pixel in After Effects":
        "After Effects のモデル設定で同じモデル単位を選び、オブジェクトスケールを 100% のままにすると、"
        "コンポジションの 1 px が After Effects でも 1 px になります",
    "Centimetres": "センチメートル",
    "Metres": "メートル",
    "Kilometres": "キロメートル",
    "Inches": "インチ",
    "Feet": "フィート",
    "Yards": "ヤード",
    "Miles": "マイル",
    "{n} units": "{n} 個の単位",
    "{layers} layer(s), {w}×{h} px at ({x}, {y}), {verts} vertices":
        "{layers} レイヤー、{w}×{h} px（位置 {x}, {y}）、頂点 {verts}",
    "Editing {n} units together": "{n} 個の単位をまとめて編集中",
    "{f:.1f} px  /  {mm:.1f} mm (36 mm film)": "{f:.1f} px  /  {mm:.1f} mm（36 mmフィルム）",
    "Depth maps (*.png *.tif *.tiff *.jpg *.npy)": "深度マップ (*.png *.tif *.tiff *.jpg *.npy)",

    # ---- dialogs and log ---------------------------------------------------
    "Open a PSD/PSB file first.": "先にPSD/PSBファイルを開いてください。",
    "Photoshop (*.psd *.psb);;All files (*)": "Photoshop (*.psd *.psb);;すべてのファイル (*)",
    "GenParallax3D project (*.gp3d)": "GenParallax3D プロジェクト (*.gp3d)",
    "glTF binary (*.glb)": "glTF バイナリ (*.glb)",
    "Open PSD/PSB": "PSD/PSBを開く",
    "Open Project": "プロジェクトを開く",
    "Untitled": "無題",
    "Loading {name}…": "{name} を読み込み中…",
    "Opening project…": "プロジェクトを開いています…",
    "Could not open PSD": "PSDを開けませんでした",
    "Could not open project": "プロジェクトを開けませんでした",
    "Could not save project": "プロジェクトを保存できませんでした",
    "Could not reload PSD": "PSDを再読み込みできませんでした",
    "Could not load depth map": "深度マップを読み込めませんでした",
    "Export failed": "書き出しに失敗しました",
    "Loaded {name}: {w}×{h}{psb}, {layers} layers, {units} reconstruction units.":
        "{name} を読み込みました：{w}×{h}{psb}、{layers} レイヤー、再構成単位 {units} 個",
    "Opened project {name}.": "プロジェクト {name} を開きました。",
    "Save Project": "プロジェクトを保存",
    "Saved project to {path}.": "プロジェクトを {path} に保存しました。",
    "Export GLB ({mode})": "GLBを書き出し（{mode}）",
    "combined": "1ファイル",
    "separate": "単位ごと",
    "Exporting…": "書き出し中…",
    "Exported {n} GLB file(s) and {meta} to {folder}.": "GLB {n} 個と {meta} を {folder} に書き出しました。",
    "Load Depth Map": "深度マップを読み込む",
    "Depth Map Convention": "深度マップの向き",
    "Is white NEAR in this depth map?\n\nYes: white = near (inverse depth, Depth Anything style)\nNo: white = far (distance)":
        "この深度マップは白が「近い」ですか？\n\nはい：白 = 近い（逆深度。Depth Anything と同じ）\nいいえ：白 = 遠い（距離）",
    "Estimating depth with {model} in a worker process (first run downloads the model)…":
        "{model} で深度を推定しています（別プロセス。初回はモデルをダウンロードします）…",
    "Depth result does not match the current document; ignored.": "深度の推定結果が現在のファイルと合わないため無視しました。",
    "Depth estimation failed: {message}": "深度推定に失敗しました：{message}",
    "Depth estimation failed": "深度推定に失敗しました",
    "{message}\n\nInstall the AI extras with:  pip install torch transformers":
        "{message}\n\nAI機能は次のコマンドで入れられます： pip install torch transformers",
    "Building depth meshes…": "深度メッシュを作成中…",
    "Applied depth map ({name}); base depth and strength taken from it.":
        "深度マップ（{name}）を適用し、基準奥行きと凹凸の強さを割り当てました。",
    "Load or estimate a depth map first.": "先に深度マップを読み込むか推定してください。",
    "Re-assigned base depth and strength from the depth map.": "深度マップから基準奥行きと凹凸の強さを割り当て直しました。",
    "No layer-order conflicts.": "レイヤー順の衝突はありません。",
    "'{name}' moved back {px} px to stay behind {uppers}.": "「{name}」を {uppers} の奥に保つため {px} px 奥へ移動しました。",
    "Updating layer order…": "レイヤー順を更新中…",
    "Layer-order update failed: {message}": "レイヤー順の更新に失敗しました：{message}",
    "⚠ Layer order not updated": "⚠ レイヤー順を更新できませんでした",
    "⚠ {n} layer-order conflict(s) — Depth ▸ Resolve Layer Order": "⚠ レイヤー順の衝突 {n} 件 — 深度 ▸ レイヤー順を整える",
    "{name}: {px} px in front of an upper layer": "{name}：上のレイヤーより {px} px 手前",
    "Layer order OK": "レイヤー順 OK",
    "The PSD changed on disk (File ▸ Reload PSD to update).": "PSDが更新されました（ファイル ▸ PSDを再読み込み で反映できます）。",
    "Reloaded {name}; settings kept for {n} units.": "{name} を再読み込みしました。{n} 個の単位の設定を引き継いでいます。",
    "Cleared depth map; units are flat.": "深度マップを消去しました。各単位は平面になります。",
    "Select layers from at least two units to merge.": "統合するには、2つ以上の単位のレイヤーを選んでください。",
    "Merged {n} units into '{name}'.": "{n} 個の単位を「{name}」に統合しました。",
    "{name}: Manual Depth uses a Custom Depth Map (falls back to the global map).":
        "{name}：手動深度には個別の深度マップを使います（未指定の場合は全体の深度マップを使います）。",
    "Switch to Free Camera, Depth or Wireframe to move the camera.":
        "カメラを動かすには「自由カメラ」「深度」「ワイヤーフレーム」のいずれかに切り替えてください。",
    "Busy — please wait until the current task finishes.": "処理中です。終わるまでお待ちください。",
    "Layer-aware 3D parallax reconstruction from PSD/PSB.": "PSD/PSBのレイヤー構造を活かした3Dパララックス再構成ツール",
    "Free Camera / Depth / Wireframe:\n"
    "  Left drag: orbit    Middle or Shift+Left drag: pan\n"
    "  Right drag / wheel: dolly    W A S D Q E: move    F: reset\n\n"
    "Click: select unit    Ctrl+Click: add / remove from selection\n"
    "1–8: display modes (Original, Reference, Free, Depth, Difference, Depth Difference, Wireframe, Parallax)\n\n"
    "Depth Difference: red = mesh nearer than the depth map, blue = further":
        "自由カメラ / 深度 / ワイヤーフレーム表示：\n"
        "  左ドラッグ：回転    中ドラッグ または Shift＋左ドラッグ：平行移動\n"
        "  右ドラッグ / ホイール：前後移動    W A S D Q E：移動    F：リセット\n\n"
        "クリック：単位を選択    Ctrl＋クリック：選択に追加 / 解除\n"
        "1〜8：表示モード（元画像、基準カメラ、自由カメラ、深度、差分、深度差分、ワイヤーフレーム、パララックス）\n\n"
        "深度差分：赤 = メッシュが深度マップより手前、青 = 奥",
}
