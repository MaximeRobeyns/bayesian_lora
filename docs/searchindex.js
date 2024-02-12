Search.setIndex({"docnames": ["bayesian_lora", "example_usage", "index", "kfac"], "filenames": ["bayesian_lora.rst", "example_usage.rst", "index.rst", "kfac.rst"], "titles": ["Bayesian Lora", "Example Usage", "Bayesian LoRA", "K-FAC Methods"], "terms": {"thi": [0, 2, 3], "file": [0, 2], "contain": [0, 2, 3], "main": [0, 2, 3], "method": [0, 2], "relat": [0, 2], "low": [0, 3], "rank": [0, 2], "adapt": [0, 3], "larg": [0, 3], "languag": [0, 2], "paper": 0, "name": [0, 3], "calcul": [0, 3], "tune": 0, "prior": 0, "network": [0, 3], "hyperparamet": 0, "paramet": [0, 3], "make": [0, 2, 3], "linearis": 0, "The": [0, 2, 3], "margin": 0, "likelihood": [0, 3], "i": [0, 2, 3], "scalar": 0, "valu": 0, "indic": [0, 3], "provid": [0, 3], "data": [0, 3], "particular": [0, 3], "A": [0, 3], "higher": 0, "consid": [0, 3], "more": [0, 3], "support": [0, 3], "under": 0, "given": 0, "bayesian_lora": [0, 2, 3], "model_evid": 0, "modul": [0, 3], "ll": 0, "tensor": [0, 3], "factor": [0, 3], "dict": [0, 3], "str": [0, 3], "tupl": [0, 3], "float": [0, 3], "l_in": [0, 3], "l_in_or_n_kfac": [0, 3], "l_out": [0, 3], "l_out_or_n_kfac": [0, 3], "n_lora": 0, "int": [0, 3], "n_kfac": [0, 3], "s2": 0, "1": [0, 3], "us": [0, 2, 3], "function": [0, 2], "instanc": 0, "varianc": 0, "your": [0, 2], "log": 0, "dataset": [0, 3], "interest": 0, "dictionari": [0, 3], "kroneck": [0, 3], "k": [0, 2], "fac": [0, 2], "return": [0, 3], "involv": 0, "two": [0, 2, 3], "step": 0, "mean": [0, 3], "For": [0, 1, 2], "first": [0, 3], "we": [0, 2, 3], "invok": 0, "admittedli": 0, "awkwardli": 0, "jacobian_mean": 0, "which": [0, 2, 3], "jacobian": 0, "respect": 0, "batch_input": 0, "batchencod": 0, "target_id": 0, "none": [0, 3], "is_s2": 0, "bool": [0, 3], "fals": [0, 3], "output_callback": 0, "callabl": [0, 3], "modeloutput": 0, "logit": [0, 3], "llm": [0, 3], "from": [0, 2, 3], "batch": [0, 3], "input": [0, 3], "exactli": [0, 3], "you": [0, 2], "would": [0, 2, 3], "pass": 0, "them": 0, "select": 0, "specif": 0, "output": [0, 3], "leav": 0, "either": 0, "wish": [0, 2], "all": [0, 2, 3], "b": [0, 3], "ar": [0, 2, 3], "an": [0, 2, 3], "post": 0, "process": 0, "whether": [0, 3], "can": [0, 2, 3], "omit": [0, 2], "take": 0, "result": 0, "kei": [0, 3], "As": 0, "see": [0, 1, 2], "wai": [0, 2, 3], "call": [0, 3], "determin": 0, "how": [0, 3], "handl": 0, "wrap": 0, "directli": [0, 2], "here": [0, 3], "assum": 0, "sequenc": 0, "default": 0, "mai": [0, 2, 3], "option": [0, 3], "want": [0, 3], "pick": 0, "out": 0, "some": [0, 2, 3], "": [0, 3], "full": [0, 2], "vocabulari": 0, "f_mu": 0, "dset": 0, "custom": 0, "callback": 0, "allow": 0, "user": 0, "forward": 0, "arbitrari": 0, "between": 0, "def": 0, "default_output_callback": 0, "cfg": 0, "els": 0, "target_logit": 0, "second": [0, 3], "precis": 0, "matrix": [0, 3], "n_logit": 0, "devic": 0, "perform": 0, "token": 0, "hf": 0, "deriv": 0, "each": [0, 3], "target": [0, 2, 3], "number": [0, 3], "e": [0, 2, 3], "g": [0, 3], "class": [0, 3], "categor": 0, "approxim": [0, 3], "kronekc": 0, "accumul": 0, "unfinish": 1, "now": [1, 3], "pleas": 1, "read": 1, "through": 1, "comment": 1, "therein": 1, "repositori": 2, "implement": [2, 3], "model": [2, 3], "simplest": [2, 3], "librari": [2, 3], "simpli": 2, "pip": 2, "If": [2, 3], "like": 2, "modifi": 2, "build": 2, "upon": 2, "while": [2, 3], "keep": 2, "separ": 2, "clone": 2, "repo": 2, "run": 2, "git": 2, "http": 2, "github": 2, "com": 2, "maximerobeyn": 2, "cd": 2, "current": [2, 3], "veri": 2, "small": 2, "ha": [2, 3], "three": 2, "core": 2, "depend": 2, "torch": [2, 3], "tqdm": [2, 3], "jaxtyp": 2, "To": 2, "end": 2, "feel": 2, "free": 2, "copi": 2, "need": 2, "own": 2, "project": 2, "start": 2, "hack": [2, 3], "There": 2, "includ": [2, 3], "befor": 2, "code": 2, "must": 2, "addit": 2, "do": 2, "after": 2, "root": 2, "plan": 2, "packag": 2, "dev": 2, "write": 2, "document": [2, 3], "requir": 2, "doc": 2, "simplic": 2, "also": 2, "just": [2, 3], "test": 2, "sure": 2, "have": [2, 3], "follow": 2, "command": 2, "onc": 2, "set": [2, 3], "up": 2, "ipython": 2, "kernel": 2, "onli": [2, 3], "so": 2, "new": [2, 3], "insid": 2, "jupyterlab": 2, "launch": 2, "conveni": 2, "lab": 2, "intern": 2, "evid": 2, "posterior": 2, "predict": [2, 3], "usag": 2, "kfac": 3, "fisher": 3, "inform": 3, "ggn": 3, "curvatur": 3, "recal": 3, "find": 3, "block": 3, "diagon": 3, "had": 3, "simpl": 3, "4": 3, "layer": 3, "eeach": 3, "mathbf": 3, "_": 3, "ell": 3, "further": 3, "product": 3, "one": 3, "correspond": 3, "activ": 3, "anoth": 3, "gradient": 3, "That": 3, "nn": 3, "index": 3, "its": 3, "approx": 3, "otim": 3, "These": 3, "around": 3, "over": 3, "mathcal": 3, "d": 3, "what": 3, "calculate_kronecker_factor": 3, "below": 3, "rather": 3, "than": 3, "numer": 3, "2": 3, "ldot": 3, "l": 3, "identifi": 3, "differ": 3, "type": 3, "t": 3, "variant": 3, "factoris": 3, "store": 3, "matric": 3, "forward_cal": 3, "ani": 3, "n_class": 3, "loader": 3, "dataload": 3, "lr_threshold": 3, "512": 3, "target_module_keyword": 3, "list": 3, "exclude_bia": 3, "use_tqdm": 3, "kronec": 3, "note": 3, "needn": 3, "lora": 3, "accept": 3, "distribut": 3, "usual": 3, "label": 3, "integ": 3, "threshold": 3, "beyond": 3, "side": 3, "length": 3, "appli": 3, "keyword": 3, "whose": 3, "hessian": 3, "particularli": 3, "when": 3, "work": 3, "By": 3, "deafult": 3, "everi": 3, "ignor": 3, "bia": 3, "term": 3, "should": 3, "show": 3, "progress": 3, "been": 3, "linear": 3, "conv1d": 3, "gpt2": 3, "sadli": 3, "exampl": 3, "fwd_call": 3, "adaptor": 3, "10": 3, "element": 3, "notic": 3, "themselv": 3, "where": 3, "4096": 3, "time": 3, "transform": 3, "abov": 3, "It": 3, "re": 3, "complet": 3, "register_hook": 3, "output_grad": 3, "100": 3, "removablehandl": 3, "regist": 3, "hook": 3, "attach": 3, "equal": 3, "featur": 3, "last": 3, "dimens": 3, "regardless": 3, "presenc": 3, "unlik": 3, "treat": 3, "turn": 3, "off": 3, "lr": 3, "wide": 3, "decid": 3, "narrow": 3, "weight": 3, "later": 3, "remov": 3, "remove_hook": 3, "save_input_hook": 3, "module_nam": 3, "has_bia": 3, "svd_dtype": 3, "dtype": 3, "float64": 3, "closur": 3, "captur": 3, "hash": 3, "portabl": 3, "map": 3, "appproxim": 3, "exce": 3, "doe": 3, "cast": 3, "svd": 3, "save_output_grad_hook": 3}, "objects": {"bayesian_lora": [[3, 0, 1, "", "calculate_kronecker_factors"]], "bayesian_lora.kfac": [[3, 0, 1, "", "register_hooks"], [3, 0, 1, "", "remove_hooks"], [3, 0, 1, "", "save_input_hook"], [3, 0, 1, "", "save_output_grad_hook"]], "bayesian_lora.main": [[0, 0, 1, "", "jacobian_mean"], [0, 0, 1, "", "model_evidence"], [0, 0, 1, "", "precision"]]}, "objtypes": {"0": "py:function"}, "objnames": {"0": ["py", "function", "Python function"]}, "titleterms": {"bayesian": [0, 2], "lora": [0, 2], "model": 0, "evid": 0, "posterior": 0, "predict": 0, "exampl": [1, 2], "usag": 1, "instal": 2, "guid": 2, "edit": 2, "hackabl": 2, "develop": 2, "jupyt": 2, "notebook": 2, "content": 2, "k": 3, "fac": 3, "method": 3, "full": 3, "rank": 3, "intern": 3, "function": 3}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.todo": 2, "sphinx": 60}, "alltitles": {"Bayesian Lora": [[0, "bayesian-lora"]], "Model Evidence": [[0, "model-evidence"]], "Posterior Predictive": [[0, "posterior-predictive"]], "Example Usage": [[1, "example-usage"]], "Bayesian LoRA": [[2, "bayesian-lora"]], "Installation Guide": [[2, "installation-guide"]], "Editable Installation": [[2, "editable-installation"]], "Hackable Installation": [[2, "hackable-installation"]], "Installation with Examples": [[2, "installation-with-examples"]], "Development Installation": [[2, "development-installation"]], "Jupyter Notebooks": [[2, "jupyter-notebooks"]], "Contents": [[2, "contents"]], "Contents:": [[2, null]], "K-FAC Methods": [[3, "k-fac-methods"]], "Full-Rank K-FAC": [[3, "full-rank-k-fac"]], "Internal Functions": [[3, "internal-functions"]]}, "indexentries": {"jacobian_mean() (in module bayesian_lora.main)": [[0, "bayesian_lora.main.jacobian_mean"]], "model_evidence() (in module bayesian_lora.main)": [[0, "bayesian_lora.main.model_evidence"]], "precision() (in module bayesian_lora.main)": [[0, "bayesian_lora.main.precision"]], "calculate_kronecker_factors() (in module bayesian_lora)": [[3, "bayesian_lora.calculate_kronecker_factors"]], "register_hooks() (in module bayesian_lora.kfac)": [[3, "bayesian_lora.kfac.register_hooks"]], "remove_hooks() (in module bayesian_lora.kfac)": [[3, "bayesian_lora.kfac.remove_hooks"]], "save_input_hook() (in module bayesian_lora.kfac)": [[3, "bayesian_lora.kfac.save_input_hook"]], "save_output_grad_hook() (in module bayesian_lora.kfac)": [[3, "bayesian_lora.kfac.save_output_grad_hook"]]}})