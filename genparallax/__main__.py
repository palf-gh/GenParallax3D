import multiprocessing

if __name__ == "__main__":
    # A frozen app re-runs its executable for spawned worker processes (AI depth
    # estimation); freeze_support() turns those runs into workers, not new windows.
    multiprocessing.freeze_support()
    from genparallax.app.main_window import main

    raise SystemExit(main())
