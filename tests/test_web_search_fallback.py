import asyncio

from app.services import web_search_service as wss


# --- host / allowlist helpers ---------------------------------------------

def test_registrable_host_strips_www_and_lowercases():
    assert wss._registrable_host("https://WWW.Fao.org/x") == "fao.org"
    assert wss._registrable_host("https://gd.eppo.int/taxon/PHYTIN") == "gd.eppo.int"
    assert wss._registrable_host("not a url") == ""


def test_host_allowed_matches_domain_and_subdomains():
    allow = ["fao.org", "europa.eu"]
    assert wss._host_allowed("https://www.fao.org/a", allow) is True
    assert wss._host_allowed("https://gd.eppo.int/x", allow) is False
    assert wss._host_allowed("https://ec.europa.eu/info", allow) is True   # subdomain
    # A look-alike suffix must not match.
    assert wss._host_allowed("https://notfao.org/x", allow) is False


# --- provider fallback chain ----------------------------------------------

class _ScriptedProvider(wss.WebSearchProvider):
    def __init__(self, name, behavior, calls):
        self.name = name
        self.behavior = behavior
        self.calls = calls

    async def search(self, query, *, max_results, allowlist):
        self.calls.append(self.name)
        if self.behavior == "error":
            raise RuntimeError("boom")
        if self.behavior == "empty":
            return []
        return [{"title": "T", "url": "https://fao.org/x", "snippet": "s", "text": "t"}]


def test_chain_falls_back_error_then_empty_then_success(monkeypatch):
    calls: list[str] = []
    behaviors = {"tavily": "error", "brave": "empty", "duckduckgo": "ok"}
    monkeypatch.setattr(wss.S, "WEB_SEARCH_PROVIDERS", "tavily,brave,duckduckgo")
    monkeypatch.setattr(wss, "_make_provider",
                        lambda name: _ScriptedProvider(name, behaviors[name], calls))

    results, served_by = asyncio.run(
        wss._chain_search("q", max_results=4, allowlist=["fao.org"])
    )
    assert served_by == "duckduckgo"
    assert calls == ["tavily", "brave", "duckduckgo"]   # strict order, all tried
    assert len(results) == 1


def test_chain_skips_unconfigured_provider(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(wss.S, "WEB_SEARCH_PROVIDERS", "tavily,duckduckgo")

    def make(name):
        if name == "tavily":
            return None  # e.g. no TAVILY_API_KEY set
        return _ScriptedProvider(name, "ok", calls)

    monkeypatch.setattr(wss, "_make_provider", make)
    results, served_by = asyncio.run(
        wss._chain_search("q", max_results=4, allowlist=["fao.org"])
    )
    assert served_by == "duckduckgo"
    assert calls == ["duckduckgo"]   # tavily skipped, never called
    assert len(results) == 1


def test_make_provider_keyed_vs_keyless(monkeypatch):
    monkeypatch.setattr(wss.S, "STAAN_API_KEY", None)
    monkeypatch.setattr(wss.S, "TAVILY_API_KEY", None)
    monkeypatch.setattr(wss.S, "BRAVE_API_KEY", None)
    # Keyed providers are unavailable (None) without their keys...
    assert wss._make_provider("staan") is None
    assert wss._make_provider("tavily") is None
    assert wss._make_provider("brave") is None
    # ...keyless DuckDuckGo is always available.
    assert isinstance(wss._make_provider("duckduckgo"), wss.DuckDuckGoProvider)
    # Once keys are present, the keyed providers instantiate.
    monkeypatch.setattr(wss.S, "STAAN_API_KEY", "staan-x")
    monkeypatch.setattr(wss.S, "TAVILY_API_KEY", "tvly-x")
    monkeypatch.setattr(wss.S, "BRAVE_API_KEY", "brave-x")
    assert isinstance(wss._make_provider("staan"), wss.StaanProvider)
    assert isinstance(wss._make_provider("tavily"), wss.TavilyProvider)
    assert isinstance(wss._make_provider("brave"), wss.BraveProvider)
    # Unknown provider names are ignored.
    assert wss._make_provider("nope") is None


def test_staan_stub_raises_so_chain_falls_back(monkeypatch):
    # A keyed-but-unwired Staan must not silently return data; it raises, and the
    # chain treats that like any provider error and advances to the next one.
    calls: list[str] = []
    monkeypatch.setattr(wss.S, "WEB_SEARCH_PROVIDERS", "staan,duckduckgo")
    monkeypatch.setattr(wss.S, "STAAN_API_KEY", "staan-x")

    def make(name):
        if name == "staan":
            return wss.StaanProvider()
        return _ScriptedProvider(name, "ok", calls)

    monkeypatch.setattr(wss, "_make_provider", make)
    results, served_by = asyncio.run(
        wss._chain_search("q", max_results=4, allowlist=["fao.org"])
    )
    assert served_by == "duckduckgo" and len(results) == 1


# --- end-to-end build (chain mocked) --------------------------------------

def _patch_chain(monkeypatch, *, text="x", results=None, served_by="tavily"):
    if results is None:
        results = [
            {"title": "FAO late blight", "url": "https://www.fao.org/page-a", "snippet": "blight", "text": text},
            {"title": "FAO duplicate domain", "url": "https://fao.org/page-b", "snippet": "dup", "text": text},
            {"title": "EPPO potato blight", "url": "https://gd.eppo.int/taxon/PHYTIN", "snippet": "potato", "text": text},
        ]

    async def _fake_chain(query, *, max_results, allowlist):
        return results, served_by

    monkeypatch.setattr(wss, "_chain_search", _fake_chain)


def test_build_contexts_numbering_dedup_and_lockstep(monkeypatch):
    long_para = ("Potato late blight is caused by Phytophthora infestans and "
                 "requires timely fungicide application and resistant varieties. ") * 3
    _patch_chain(monkeypatch, text=long_para)

    contexts, sources = asyncio.run(
        wss.web_search_and_build_contexts(
            "potato late blight", max_results=4, max_chars=10000, sid_offset=2
        )
    )

    # Two source domains survive the per-domain dedup (fao.org, eppo.int).
    assert len(contexts) == len(sources) == 2
    # SID numbering continues from the offset (2 KO sources already present).
    assert [s.sid for s in sources] == ["S3", "S4"]
    assert contexts[0].startswith("[S3]")
    assert contexts[1].startswith("[S4]")
    # Provenance is recorded as the registrable domain.
    assert sources[0].project == "fao.org"
    assert sources[1].project == "gd.eppo.int"
    assert sources[0].url == "https://www.fao.org/page-a"


def test_build_contexts_respects_char_budget(monkeypatch):
    _patch_chain(monkeypatch, text="short")
    contexts, sources = asyncio.run(
        wss.web_search_and_build_contexts("x", max_results=4, max_chars=5, sid_offset=0)
    )
    assert contexts == [] and sources == []


def test_non_positive_budget_short_circuits(monkeypatch):
    # The chain must not even be consulted when there's no room.
    called = {"hit": False}

    async def _boom(query, *, max_results, allowlist):
        called["hit"] = True
        return [], None

    monkeypatch.setattr(wss, "_chain_search", _boom)
    contexts, sources = asyncio.run(
        wss.web_search_and_build_contexts("x", max_results=4, max_chars=0)
    )
    assert contexts == [] and sources == [] and called["hit"] is False
