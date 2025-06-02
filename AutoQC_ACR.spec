# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = [
    *collect_dynamic_libs("pandas"),
    *collect_dynamic_libs("numpy"),
]

a = Analysis(  # type: ignore
    ["src/main.py"],
    pathex=["src/backend/smaaf"],
    binaries=binaries,
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)  # type: ignore

splash = Splash("src/frontend/assets/splash.png", a.binaries, a.datas)  # type: ignore

exe = EXE(  # type: ignore
    pyz,
    a.scripts,
    splash,
    [],
    exclude_binaries=True,
    name="AutoQC_ACR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(  # type: ignore
    exe,
    splash.binaries,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=".",
)
