# -*- mode: python ; coding: utf-8 -*-

"""AutoQC_ACR.spec

This is a .spec file used to control the behaviour of pyinstaller when building a distributable .exe directory.
Relevant binaries are collected and bundled, a splash screen is added and python code is converted to .pyc

"""

from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = [
    *collect_dynamic_libs("pandas"),
    *collect_dynamic_libs("numpy"),
]

datas = [("src/backend/assets", "assets")]

a = Analysis(  # type: ignore
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},

    runtime_hooks=[],
    excludes=['PyQt5','PySide2','PySide6'],
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
