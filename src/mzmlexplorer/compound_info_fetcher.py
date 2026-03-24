"""Utilities for fetching compound information from online databases (PubChem)."""

import json
import pickle
import re
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
REQUEST_DELAY = 0.25  # seconds between batch/individual HTTP calls
CACHE_MAX_AGE_DAYS = 90
_BATCH_CHUNK_SIZE = 100  # max CIDs per batch request

# Regex for CAS Registry Numbers (e.g. "50-78-2", "123456-78-9")
_CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")

# ---------------------------------------------------------------------------
# Persistent cache
# ---------------------------------------------------------------------------
# Location: ~/.mzmlexplorer/http_cache.pkl
# Entry format: {key: {"data": <any>, "fetched_at": datetime}}
#
# Two types of keys are stored in the same dict:
#   • URL strings  – used for individual identifier→CID lookups
#   • "pubchem:props:{cid}" / "pubchem:syns:{cid}" – per-CID property / synonym data
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".mzmlexplorer"
_CACHE_FILE = _CACHE_DIR / "http_cache.pkl"
_cache: Optional[dict] = None  # loaded lazily


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "rb") as fh:
                _cache = pickle.load(fh)
                if not isinstance(_cache, dict):
                    _cache = {}
        except Exception:
            _cache = {}
    else:
        _cache = {}
    return _cache


def _save_cache() -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "wb") as fh:
            pickle.dump(_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass  # cache write failures are non-fatal


def _cache_get(key: str) -> tuple[bool, object]:
    """Return ``(hit, data)``. *hit* is False when the entry is missing or stale."""
    cache = _load_cache()
    entry = cache.get(key)
    if entry is None:
        return False, None
    fetched_at: datetime = entry.get("fetched_at", datetime.min)
    if datetime.now() - fetched_at > timedelta(days=CACHE_MAX_AGE_DAYS):
        return False, None  # stale – re-fetch
    return True, entry.get("data")


def _cache_set(key: str, data: object) -> None:
    cache = _load_cache()
    cache[key] = {"data": data, "fetched_at": datetime.now()}
    _save_cache()


def _cache_is_fresh(key: str) -> bool:
    hit, _ = _cache_get(key)
    return hit


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_get_raw(url: str) -> Optional[dict]:
    """Plain HTTP GET → parsed JSON or None.  No caching."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "mzMLExplorer/1.0 (scientific research tool)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _pubchem_get(url: str) -> Optional[dict]:
    """GET a PubChem URL, result cached by URL key.

    Used for individual identifier→CID lookups whose URL is unique per
    identifier and therefore safe to cache verbatim.
    """
    hit, cached_data = _cache_get(url)
    if hit:
        return cached_data
    result = _http_get_raw(url)
    _cache_set(url, result)
    return result


# ---------------------------------------------------------------------------
# CID resolution (one HTTP request per identifier, URL-cached)
# ---------------------------------------------------------------------------


def _get_cid_by_identifier(identifier: str) -> Optional[int]:
    """Return the first PubChem CID matching *identifier* (name, CAS, etc.)."""
    url = f"{PUBCHEM_BASE_URL}/compound/name/{urllib.parse.quote(identifier.strip())}/cids/JSON"
    data = _pubchem_get(url)
    time.sleep(REQUEST_DELAY)
    if data and "IdentifierList" in data:
        cids = data["IdentifierList"].get("CID", [])
        if cids:
            return cids[0]
    return None


# ---------------------------------------------------------------------------
# Per-CID cache key helpers
# ---------------------------------------------------------------------------


def _props_key(cid: int) -> str:
    return f"pubchem:props:{cid}"


def _syns_key(cid: int) -> str:
    return f"pubchem:syns:{cid}"


# ---------------------------------------------------------------------------
# Batch property / synonym fetchers
# ---------------------------------------------------------------------------


def _batch_fetch_properties(cids: list) -> None:
    """Fetch properties for *cids* using comma-separated CID requests.

    Results are stored individually under ``pubchem:props:{cid}`` so that
    subsequent calls for any subset of those CIDs are served from cache.
    Requests are chunked at *_BATCH_CHUNK_SIZE* CIDs each.
    """
    for i in range(0, len(cids), _BATCH_CHUNK_SIZE):
        chunk = cids[i : i + _BATCH_CHUNK_SIZE]
        cid_str = ",".join(str(c) for c in chunk)
        url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid_str}/property/MolecularFormula,CanonicalSMILES,IUPACName,InChI,InChIKey,XLogP,Charge,LiteratureCount,Fingerprint2D,Title/JSON"
        data = _http_get_raw(url)
        time.sleep(REQUEST_DELAY)
        fetched: set = set()
        if data and "PropertyTable" in data:
            for props in data["PropertyTable"].get("Properties", []):
                cid = props.get("CID")
                if cid is not None:
                    _cache_set(_props_key(cid), props)
                    fetched.add(cid)
        # Mark CIDs with no data so they are not re-requested until the cache expires
        for c in chunk:
            if c not in fetched:
                _cache_set(_props_key(c), None)


def _batch_fetch_synonyms(cids: list) -> None:
    """Fetch synonyms for *cids* using comma-separated CID requests.

    Results are stored individually under ``pubchem:syns:{cid}``.
    Requests are chunked at *_BATCH_CHUNK_SIZE* CIDs each.
    """
    for i in range(0, len(cids), _BATCH_CHUNK_SIZE):
        chunk = cids[i : i + _BATCH_CHUNK_SIZE]
        cid_str = ",".join(str(c) for c in chunk)
        url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid_str}/synonyms/JSON"
        data = _http_get_raw(url)
        time.sleep(REQUEST_DELAY)
        fetched: set = set()
        if data and "InformationList" in data:
            for info in data["InformationList"].get("Information", []):
                cid = info.get("CID")
                if cid is not None:
                    _cache_set(_syns_key(cid), info.get("Synonym", []))
                    fetched.add(cid)
        for c in chunk:
            if c not in fetched:
                _cache_set(_syns_key(c), [])


# ---------------------------------------------------------------------------
# CAS Common Chemistry API
# ---------------------------------------------------------------------------
# Base URL: https://commonchemistry.cas.org/api
# Endpoints:
#   /detail?cas_rn={rn}  – full record for a known CAS RN
#   /search?q={query}    – search by name / formula / InChI etc.
# Authentication via X-API-Key request header.
# Cache keys: "cas:detail:{cas_rn}"  and  "cas:search:{identifier}"
# ---------------------------------------------------------------------------

CAS_API_BASE = "https://commonchemistry.cas.org/api"


def load_cas_api_key() -> Optional[str]:
    """Load the CAS API key from *secrets.json*.

    Searches in order:
    1. Project root – 3 levels above this module (``secrets.json`` next to pyproject.toml).
    2. ``~/.mzmlexplorer/secrets.json`` – per-user fallback.

    Returns the key string, or ``None`` when no key is found.
    """
    candidates = [
        Path(__file__).parent.parent.parent / "secrets.json",
        Path.home() / ".mzmlexplorer" / "secrets.json",
    ]
    for path in candidates:
        try:
            if path.exists():
                with open(path) as fh:
                    data = json.load(fh)
                    key = data.get("CAS_API_KEY")
                    if key and str(key).strip():
                        return str(key).strip()
        except Exception:
            continue
    return None


def _cas_detail_key(cas_rn: str) -> str:
    return f"cas:detail:{cas_rn}"


def _cas_search_key(identifier: str) -> str:
    return f"cas:search:{identifier.lower().strip()}"


def _http_get_cas(url: str, api_key: str) -> Optional[dict]:
    """HTTP GET to the CAS Common Chemistry API with *api_key* authentication."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "mzMLExplorer/1.0 (scientific research tool)",
                "X-API-Key": api_key,
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _fetch_and_cache_cas_detail(cas_rn: str, api_key: str) -> Optional[dict]:
    """Fetch (or return cached) CAS Common Chemistry detail for *cas_rn*."""
    key = _cas_detail_key(cas_rn)
    hit, cached = _cache_get(key)
    if hit:
        return cached
    url = f"{CAS_API_BASE}/detail?cas_rn={urllib.parse.quote(cas_rn.strip())}"
    data = _http_get_cas(url, api_key)
    time.sleep(REQUEST_DELAY)
    _cache_set(key, data)
    return data


def _search_cas_for_rn(name: str, api_key: str) -> Optional[str]:
    """Search CAS by *name* and return the first matched CAS RN, or ``None``."""
    key = _cas_search_key(name)
    hit, cached_rn = _cache_get(key)
    if hit:
        return cached_rn
    url = f"{CAS_API_BASE}/search?q={urllib.parse.quote(name.strip())}"
    data = _http_get_cas(url, api_key)
    time.sleep(REQUEST_DELAY)
    rn: Optional[str] = None
    if data and "results" in data:
        results = data["results"]
        if results:
            rn = results[0].get("rn")
    _cache_set(key, rn)
    return rn


# ---------------------------------------------------------------------------
# Result assembly from per-CID cache
# ---------------------------------------------------------------------------


def _assemble_result(cid: int, original_cas: Optional[str]) -> dict:
    """Build a result dict for *cid* entirely from cached per-CID data."""
    result: dict = {"cid": str(cid), "synonyms": [], "cas_number": None}

    _, props = _cache_get(_props_key(cid))
    if props:
        result["smiles"] = props.get("ConnectivitySMILES")
        if not result["smiles"]:
            result["smiles"] = props.get("CanonicalSMILES")
        result["molecular_formula"] = props.get("MolecularFormula")
        result["iupac_name"] = props.get("IUPACName")
        result["inchi"] = props.get("InChI")
        result["inchikey"] = props.get("InChIKey")
        result["xlogp"] = props.get("XLogP")
        result["charge"] = props.get("Charge")
        result["literature_count"] = props.get("LiteratureCount")
        result["fingerprint2d"] = props.get("Fingerprint2D")
        result["title"] = props.get("Title")

    _, synonyms = _cache_get(_syns_key(cid))
    if synonyms:
        result["synonyms"] = synonyms[:30]
        if not original_cas:
            for syn in result["synonyms"]:
                if _CAS_PATTERN.match(str(syn)):
                    result["cas_number"] = syn
                    break

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_pubchem_info_batch(
    compounds: list,
    progress_callback: Optional[Callable] = None,
    cas_api_key: Optional[str] = None,
) -> list:
    """Fetch PubChem information for a list of compounds using batched HTTP requests.

    Parameters
    ----------
    compounds:
        List of ``(name, cas)`` tuples.  *cas* may be ``None``.
    progress_callback:
        Optional callable ``(i, total, name) -> bool``.  Called after each CID
        resolution step (still sequential since each needs its own identifier
        lookup).  Return ``False`` to cancel; remaining CIDs will be ``None``
        and the batch network requests will only be made for already-resolved CIDs.

    Returns
    -------
    List of result dicts (same schema as ``fetch_pubchem_info``), one per input
    entry, with ``None`` for compounds that could not be resolved.

    Network efficiency
    ------------------
    After CID resolution, properties and synonyms for **all** resolved CIDs that
    are not already cached are fetched in a single HTTP request each (chunked at
    ``_BATCH_CHUNK_SIZE`` CIDs), greatly reducing network pressure for large lists.
    When *cas_api_key* is supplied, each resolved compound is additionally
    enriched with data from the CAS Common Chemistry API (cached per CAS RN).
    """
    # Step 1 — resolve CIDs (individual, URL-cached)
    cids: list = []
    cancelled = False
    for i, (name, cas) in enumerate(compounds):
        cid = None
        if not cancelled:
            if cas:
                cid = _get_cid_by_identifier(cas.strip())
            if cid is None and name:
                cid = _get_cid_by_identifier(name.strip())
        cids.append(cid)

        if progress_callback is not None:
            if not progress_callback(i, len(compounds), name or ""):
                cancelled = True  # fill remaining cids with None; batch fetch skipped for them

    # Step 2 — batch-fetch properties and synonyms for uncached resolved CIDs
    resolved = list(dict.fromkeys(c for c in cids if c is not None))
    needs_props = [c for c in resolved if not _cache_is_fresh(_props_key(c))]
    needs_syns = [c for c in resolved if not _cache_is_fresh(_syns_key(c))]

    if needs_props:
        _batch_fetch_properties(needs_props)
    if needs_syns:
        _batch_fetch_synonyms(needs_syns)

    # Step 3 — assemble PubChem results from per-CID cache
    results = [None if cid is None else _assemble_result(cid, cas) for (name, cas), cid in zip(compounds, cids)]

    # Step 4 — enrich with CAS Common Chemistry data (only when API key present)
    if cas_api_key:
        for (name, cas), result in zip(compounds, results):
            if result is None:
                continue
            # Prefer the CAS RN already resolved from PubChem synonyms; fall back
            # to a CAS name search when none is available.
            cas_rn = result.get("cas_number")
            if not cas_rn and name:
                cas_rn = _search_cas_for_rn(name, cas_api_key)
            if not cas_rn:
                continue
            cas_detail = _fetch_and_cache_cas_detail(cas_rn, cas_api_key)
            if not cas_detail:
                continue
            result["cas_preferred_name"] = cas_detail.get("name") or None
            result["cas_rn_confirmed"] = cas_detail.get("rn") or None
            exp_props = cas_detail.get("experimentalProperties") or []
            result["cas_experimental_properties"] = json.dumps(exp_props, ensure_ascii=False) if exp_props else None

    return results


def fetch_pubchem_info(name: str, cas: Optional[str] = None, cas_api_key: Optional[str] = None) -> Optional[dict]:
    """Fetch PubChem (and optionally CAS) information for a single compound.

    Delegates to ``fetch_pubchem_info_batch`` for a single-element list so that
    all caching and HTTP logic is shared.

    Returns a dict (or None if nothing was found) with keys:

    ``cid``                         – PubChem Compound ID (str)
    ``smiles``                      – canonical SMILES string
    ``molecular_formula``           – molecular formula
    ``iupac_name``                  – IUPAC name
    ``inchi``                       – InChI string
    ``inchikey``                    – InChIKey
    ``xlogp``                       – XLogP3 lipophilicity value (float or None)
    ``charge``                      – formal charge (int or None)
    ``literature_count``            – PubMed literature count (int or None)
    ``fingerprint2d``               – base64-encoded 2-D fingerprint string (or None)
    ``title``                       – PubChem preferred title / name
    ``synonyms``                    – list[str], up to 30 synonyms
    ``cas_number``                  – CAS extracted from synonyms (str or None)
    ``cas_preferred_name``          – CAS preferred name (only when *cas_api_key* given)
    ``cas_rn_confirmed``            – CAS RN confirmed by CAS API (or None)
    ``cas_experimental_properties`` – JSON array of experimental property objects (or None)
    """
    return fetch_pubchem_info_batch([(name, cas)], cas_api_key=cas_api_key)[0]


# Regex for CAS Registry Numbers (e.g. "50-78-2", "123456-78-9")
_CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")

# ---------------------------------------------------------------------------
# HTTP response cache
# ---------------------------------------------------------------------------
# Cache file location: ~/.mzmlexplorer/http_cache.pkl
# Structure: {url: {"data": <parsed JSON or None>, "fetched_at": datetime}}
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".mzmlexplorer"
_CACHE_FILE = _CACHE_DIR / "http_cache.pkl"
_cache: Optional[dict] = None  # loaded lazily


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "rb") as fh:
                _cache = pickle.load(fh)
                if not isinstance(_cache, dict):
                    _cache = {}
        except Exception:
            _cache = {}
    else:
        _cache = {}
    return _cache


def _save_cache() -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "wb") as fh:
            pickle.dump(_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass  # cache write failures are non-fatal


def _cache_get(url: str) -> tuple[bool, Optional[dict]]:
    """Return ``(hit, data)``. *hit* is False when a fresh fetch is needed."""
    cache = _load_cache()
    entry = cache.get(url)
    if entry is None:
        return False, None
    fetched_at: datetime = entry.get("fetched_at", datetime.min)
    if datetime.now() - fetched_at > timedelta(days=CACHE_MAX_AGE_DAYS):
        return False, None  # stale
    return True, entry.get("data")


def _cache_set(url: str, data: Optional[dict]) -> None:
    cache = _load_cache()
    cache[url] = {"data": data, "fetched_at": datetime.now()}
    _save_cache()


def _pubchem_get(url: str) -> Optional[dict]:
    """GET a PubChem URL, return parsed JSON dict or None on any error.

    Results are persisted in a local pickle cache keyed by URL.
    Entries older than ``CACHE_MAX_AGE_DAYS`` days are re-fetched automatically.
    """
    hit, cached_data = _cache_get(url)
    if hit:
        return cached_data

    # Not in cache (or stale) — perform the actual HTTP request
    result: Optional[dict] = None
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "mzMLExplorer/1.0 (scientific research tool)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass  # result stays None

    _cache_set(url, result)
    return result


def _get_cid_by_identifier(identifier: str) -> Optional[int]:
    """Return the first PubChem CID matching *identifier* (name, CAS, etc.)."""
    url = f"{PUBCHEM_BASE_URL}/compound/name/{urllib.parse.quote(identifier.strip())}/cids/JSON"
    data = _pubchem_get(url)
    time.sleep(REQUEST_DELAY)
    if data and "IdentifierList" in data:
        cids = data["IdentifierList"].get("CID", [])
        if cids:
            return cids[0]
    return None


def fetch_pubchem_info(name: str, cas: Optional[str] = None) -> Optional[dict]:
    """
    Fetch compound information from PubChem for a given compound name and/or
    CAS registry number.

    The lookup order is:
    1. CAS number (if provided and non-empty)
    2. Compound name

    Returns a dict (or None if nothing was found) with keys:

    ``cid``               – PubChem Compound ID (str)
    ``canonical_smiles``  – canonical SMILES string
    ``molecular_formula`` – molecular formula
    ``iupac_name``        – IUPAC name
    ``inchi``             – InChI string
    ``inchikey``          – InChIKey
    ``synonyms``          – list[str], up to 30 synonyms
    ``cas_number``        – CAS extracted from synonyms (str or None)
    """
    cid: Optional[int] = None

    # 1. Try CAS number
    if cas:
        cas_clean = cas.strip()
        if cas_clean:
            cid = _get_cid_by_identifier(cas_clean)

    # 2. Fall back to compound name
    if cid is None and name:
        cid = _get_cid_by_identifier(name.strip())

    if cid is None:
        return None

    # Fetch properties
    props_url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/property/MolecularFormula,CanonicalSMILES,IUPACName,InChI,InChIKey/JSON"
    props_data = _pubchem_get(props_url)
    time.sleep(REQUEST_DELAY)

    result: dict = {
        "cid": str(cid),
        "synonyms": [],
        "cas_number": None,
    }

    if props_data and "PropertyTable" in props_data:
        props = props_data["PropertyTable"].get("Properties", [{}])[0]
        result["smiles"] = props.get("ConnectivitySMILES")  # Returned by PubChem for some compounds, but not all. Fallback to CanonicalSMILES if missing.
        if not result["smiles"]:
            result["smiles"] = props.get("CanonicalSMILES")
        result["molecular_formula"] = props.get("MolecularFormula")
        result["iupac_name"] = props.get("IUPACName")
        result["inchi"] = props.get("InChI")
        result["inchikey"] = props.get("InChIKey")

    # Fetch synonyms (used for alternative names + CAS extraction)
    syn_url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/synonyms/JSON"
    syn_data = _pubchem_get(syn_url)
    time.sleep(REQUEST_DELAY)

    if syn_data and "InformationList" in syn_data:
        info_list = syn_data["InformationList"].get("Information", [{}])
        if info_list:
            synonyms = info_list[0].get("Synonym", [])
            result["synonyms"] = synonyms[:30]

            # Extract CAS from synonym list when not already supplied by the caller
            if not cas:
                for syn in result["synonyms"]:
                    if _CAS_PATTERN.match(str(syn)):
                        result["cas_number"] = syn
                        break

    return result
