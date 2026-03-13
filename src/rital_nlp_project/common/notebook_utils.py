def init_notebook(seaborn_theme="darkgrid"):
    try:
        import seaborn as sns
        sns.set_theme(style=seaborn_theme)
    except Exception:
        pass
    try:
        from IPython import get_ipython
        ip = get_ipython()
        ip.run_line_magic("load_ext", "autoreload")
        ip.run_line_magic("autoreload", "2")
    except Exception:
        pass
