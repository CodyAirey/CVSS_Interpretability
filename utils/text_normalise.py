import re

RE_CVE     = re.compile(r'\bCVE-\d{4}-\d{4,7}\b', re.IGNORECASE)
RE_IPPORT  = re.compile(r'\b(?:(?:\d{1,3}\.){3}\d{1,3}|\[[A-Fa-f0-9:]+\]):\d{2,5}\b')
RE_IPV4    = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
RE_IPV6    = re.compile(r'\b(?:[A-Fa-f0-9]{0,4}:){2,7}[A-Fa-f0-9]{0,4}\b')
RE_VER     = re.compile(r'\bv?\d+(?:\.\d+){1,3}(?:-[A-Za-z0-9]+)?\b')

# Paths: absolute and relative (don’t replace function calls like foo(…))
RE_PATH_WIN_ABS  = re.compile(r'\b[A-Za-z]:\\[^\s]+')
RE_PATH_UNIX_ABS = re.compile(r'(?<!\w)/[^\s]+')
RE_PATH_REL      = re.compile(r'(?<!\w)(?:\.{1,2}/|~/?|[A-Za-z0-9._-]+/)[A-Za-z0-9._-]+(?:[/\\][A-Za-z0-9._-]+)*')

def _mask_paths(text: str) -> str:
    def repl(rx, s):
        out, last = [], 0
        for m in rx.finditer(s):
            a, b = m.span()
            # if immediately followed by '(' treat it as a function call prefix; skip
            if s[b:b+1] == '(':
                continue
            out.append(s[last:a]); out.append("<PATH>"); last = b
        out.append(s[last:])
        return "".join(out)
    t = repl(RE_PATH_WIN_ABS, text)
    t = repl(RE_PATH_UNIX_ABS, t)
    t = repl(RE_PATH_REL, t)
    return t

def normalise_text(s: str, enabled: bool = True) -> str:
    if not enabled:
        return s
    t = s
    t = RE_IPPORT.sub("<IPPORT>", t)
    t = RE_IPV4.sub("<IP>", t)
    t = RE_IPV6.sub("<IP>", t)
    t = RE_CVE.sub("<CVE>", t)
    t = RE_VER.sub("<VER>", t)
    t = _mask_paths(t)
    return t
