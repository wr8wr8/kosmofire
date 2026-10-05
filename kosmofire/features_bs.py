import numpy as np
from scipy import ndimage as ndi

from .config import REFLECTANCE_SCALE, SAR_DB_SCALE, SCL_CLOUD_CLASSES
from .landcover import UNKNOWN_INDEX, landcover_index, landcover_one_hot, one_hot_names
from .unmixing import unmix

EPS = 1e-6
CONTEXT_SIZES = (5, 15, 41)
PHENO_BACKGROUND_DNBR = 0.10
PHENO_FULL_SUPPORT_PIXELS = 3000
DEFAULT_GROUPS = ("extra", "sar_texture", "unmix", "cloud", "indices2", "pheno")

BASE_NAMES = [
    "nbr_pre", "nbr_post", "dnbr", "rdnbr", "ndvi_pre", "ndvi_post", "dndvi",
    "b12_post", "b8a_post", "b8a_pre", "b12_pre", "b11_post", "b4_post", "b2_post", "b3_post",
    "d_b8a", "d_b12", "d_b11", "d_b4", "d_b2", "ndmi_pre", "ndmi_post", "dndmi", "bai_post", "dbai",
    "rededge_pre", "rededge_post", "d_rededge",
    "vv_pre", "vh_pre", "vv_post", "vh_post", "dvv", "dvh", "ratio_pre", "ratio_post", "dratio",
    "cloud_pre", "cloud_post", "shadow_or_cloud",
    "dem", "slope", "dnbr_rank_local",
]
EXTRA_NAMES = ["dnbr_minus_dndvi", "mirbi_pre", "mirbi_post", "dmirbi", "savi_pre", "savi_post", "dsavi"]
SAR_TEXTURE_NAMES = ["vh_var_pre", "vh_var_post", "dvh_var", "vv_var_pre", "vv_var_post", "dvv_var"]
UNMIX_NAMES = [
    "frac_char_pre", "frac_veg_pre", "frac_soil_pre",
    "frac_char_post", "frac_veg_post", "frac_soil_post",
    "d_frac_char", "d_frac_veg", "d_frac_soil",
]
CLOUD_NAMES = ["cloud_local_31"]
INDICES2_NAMES = ["bais2_pre", "bais2_post", "dbais2", "kndvi_pre", "kndvi_post", "dkndvi"]
PHENO_NAMES = ["dnbr_corr", "delta_background", "background_pixels"]
SAR_ONLY_PREFIXES = ("vv_", "vh_", "dvv", "dvh", "ratio_", "dratio", "dem", "slope", "landcover", "lc_", "cloud_local")


def sar_only_indices(names: list[str]) -> list[int]:
    return [i for i, name in enumerate(names) if name.startswith(SAR_ONLY_PREFIXES)]


def bs_feature_names(groups: tuple[str, ...] = DEFAULT_GROUPS) -> list[str]:
    names = list(BASE_NAMES)
    for size in CONTEXT_SIZES:
        names += [f"dnbr_mean_{size}", f"dnbr_std_{size}", f"dndvi_mean_{size}", f"dvh_mean_{size}", f"b12_post_mean_{size}", f"dnbr_support_{size}"]
    names += ["dnbr_median_5", "dnbr_max_15", "dnbr_p_burn_share_15"]
    names += one_hot_names() if "onehot" in groups else ["landcover"]
    names += ["chip_dnbr_p90", "chip_dnbr_p50", "dnbr_minus_chip_p50", "dnbr_over_chip_p90"]
    for group, block in (("extra", EXTRA_NAMES), ("sar_texture", SAR_TEXTURE_NAMES), ("unmix", UNMIX_NAMES), ("cloud", CLOUD_NAMES), ("indices2", INDICES2_NAMES), ("pheno", PHENO_NAMES)):
        if group in groups:
            names += block
    return names


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + EPS)


def _local_variance(band: np.ndarray, size: int) -> np.ndarray:
    mean = ndi.uniform_filter(band, size=size)
    sq = ndi.uniform_filter(band * band, size=size)
    return np.maximum(sq - mean * mean, 0.0)


def _cloud_mask(scl: np.ndarray) -> np.ndarray:
    out = np.zeros(scl.shape, dtype=bool)
    for c in SCL_CLOUD_CLASSES:
        out |= scl == c
    return out


def masked_window_stats(values: np.ndarray, valid: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight = valid.astype(np.float32)
    filled = np.where(valid, values, 0.0).astype(np.float32)
    support = ndi.uniform_filter(weight, size=size)
    total = ndi.uniform_filter(filled, size=size)
    total_sq = ndi.uniform_filter(filled * filled, size=size)
    present = support > 0.5 / (size * size)
    safe = np.where(present, support, 1.0)
    mean = np.where(present, total / safe, np.nan)
    var = np.where(present, np.maximum(total_sq / safe - (total / safe) ** 2, 0.0), np.nan)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32), support.astype(np.float32)


def masked_window_mean(values: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    return masked_window_stats(values, valid, size)[0]


def phenological_offset(dnbr: np.ndarray, valid: np.ndarray, mean15: np.ndarray, landcover_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    background = valid & np.isfinite(dnbr) & (dnbr < PHENO_BACKGROUND_DNBR) & np.isfinite(mean15) & (mean15 < PHENO_BACKGROUND_DNBR)
    classes = int(UNKNOWN_INDEX) + 1
    counts = np.bincount(landcover_idx[background].astype(np.int64), minlength=classes).astype(np.float64)
    sums = np.bincount(landcover_idx[background].astype(np.int64), weights=dnbr[background].astype(np.float64), minlength=classes)
    means = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
    offsets = means * np.minimum(1.0, counts / PHENO_FULL_SUPPORT_PIXELS)
    index = landcover_idx.astype(np.int64)
    delta = offsets[index].astype(np.float32)
    return (dnbr - delta).astype(np.float32), delta, counts[index].astype(np.float32)


def estimate_kndvi_sigma(reflectance_pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    values = []
    for b8a, b4 in reflectance_pairs:
        values.append(np.abs(b8a - b4).ravel()[::97])
    return float(np.median(np.concatenate(values)))


def bs_features(
    s2_pre: np.ndarray,
    s2_post: np.ndarray,
    s1_pre: np.ndarray,
    s1_post: np.ndarray,
    aux: np.ndarray,
    groups: tuple[str, ...] = DEFAULT_GROUPS,
    endmembers: np.ndarray | None = None,
    kndvi_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pre = s2_pre[:9].astype(np.float32) / REFLECTANCE_SCALE
    post = s2_post[:9].astype(np.float32) / REFLECTANCE_SCALE
    b2p, b3p, b4p, b5p, b6p, b7p, b8ap, b11p, b12p = pre
    b2q, b3q, b4q, b5q, b6q, b7q, b8aq, b11q, b12q = post

    nbr_pre = _nd(b8ap, b12p)
    nbr_post = _nd(b8aq, b12q)
    dnbr = nbr_pre - nbr_post
    rdnbr = dnbr / np.sqrt(np.abs(nbr_pre) + 0.001)
    ndvi_pre = _nd(b8ap, b4p)
    ndvi_post = _nd(b8aq, b4q)
    dndvi = ndvi_pre - ndvi_post
    ndmi_pre = _nd(b8ap, b11p)
    ndmi_post = _nd(b8aq, b11q)
    bai_post = np.clip(1.0 / ((0.1 - b4q) ** 2 + (0.06 - b8aq) ** 2 + EPS), 0, 500)
    bai_pre = np.clip(1.0 / ((0.1 - b4p) ** 2 + (0.06 - b8ap) ** 2 + EPS), 0, 500)
    red_pre = _nd(b7p, b5p)
    red_post = _nd(b7q, b5q)

    vv_pre, vh_pre = (ndi.median_filter(s1_pre[i].astype(np.float32) / SAR_DB_SCALE, size=3) for i in range(2))
    vv_post, vh_post = (ndi.median_filter(s1_post[i].astype(np.float32) / SAR_DB_SCALE, size=3) for i in range(2))
    dvh = vh_post - vh_pre
    ratio_pre = vh_pre - vv_pre
    ratio_post = vh_post - vv_post

    cloud_pre = _cloud_mask(s2_pre[9])
    cloud_post = _cloud_mask(s2_post[9])
    scl_bad = cloud_pre | cloud_post
    valid = ~scl_bad & np.isfinite(dnbr)

    dem = aux[0].astype(np.float32)
    slope = aux[1].astype(np.float32)
    landcover_idx = landcover_index(aux[2])

    mean31, _, _ = masked_window_stats(dnbr, valid, 31)
    layers = [
        nbr_pre, nbr_post, dnbr, rdnbr, ndvi_pre, ndvi_post, dndvi,
        b12q, b8aq, b8ap, b12p, b11q, b4q, b2q, b3q,
        b8ap - b8aq, b12p - b12q, b11p - b11q, b4p - b4q, b2p - b2q,
        ndmi_pre, ndmi_post, ndmi_pre - ndmi_post, bai_post, bai_post - bai_pre,
        red_pre, red_post, red_pre - red_post,
        vv_pre, vh_pre, vv_post, vh_post, vv_post - vv_pre, dvh, ratio_pre, ratio_post, ratio_post - ratio_pre,
        cloud_pre.astype(np.float32), cloud_post.astype(np.float32), scl_bad.astype(np.float32),
        dem, slope, dnbr - mean31,
    ]

    mean15 = None
    for size in CONTEXT_SIZES:
        mean, std, support = masked_window_stats(dnbr, valid, size)
        if size == 15:
            mean15 = mean
        layers += [
            mean,
            std,
            masked_window_mean(dndvi, valid, size),
            ndi.uniform_filter(dvh, size=size),
            masked_window_mean(b12q, valid, size),
            support,
        ]
    mean5 = masked_window_mean(dnbr, valid, 5)
    imputed = np.where(valid, dnbr, np.where(np.isfinite(mean5), mean5, 0.0)).astype(np.float32)
    layers.append(ndi.median_filter(imputed, size=5))
    neg_inf = np.float32(-1e9)
    window_max = ndi.maximum_filter(np.where(valid, dnbr, neg_inf), size=15)
    layers.append(np.where(window_max > -1e8, window_max, np.nan).astype(np.float32))
    layers.append(masked_window_mean((dnbr > 0.15).astype(np.float32), valid, 15))
    if "onehot" in groups:
        layers += list(landcover_one_hot(aux[2]))
    else:
        layers.append(landcover_idx)

    finite_valid = dnbr[valid]
    if finite_valid.size:
        p50, p90 = np.percentile(finite_valid, [50, 90])
    else:
        p50, p90 = 0.0, 0.0
    ones = np.ones_like(dnbr)
    layers += [ones * p90, ones * p50, dnbr - p50, dnbr / (abs(p90) + 0.05)]

    if "extra" in groups:
        mirbi_pre = 10.0 * b12p - 9.8 * b11p + 2.0
        mirbi_post = 10.0 * b12q - 9.8 * b11q + 2.0
        savi_pre = 1.5 * (b8ap - b4p) / (b8ap + b4p + 0.5)
        savi_post = 1.5 * (b8aq - b4q) / (b8aq + b4q + 0.5)
        layers += [dnbr - dndvi, mirbi_pre, mirbi_post, mirbi_pre - mirbi_post, savi_pre, savi_post, savi_pre - savi_post]
    if "sar_texture" in groups:
        vh_var_pre = _local_variance(vh_pre, 7)
        vh_var_post = _local_variance(vh_post, 7)
        vv_var_pre = _local_variance(vv_pre, 7)
        vv_var_post = _local_variance(vv_post, 7)
        layers += [vh_var_pre, vh_var_post, vh_var_pre - vh_var_post, vv_var_pre, vv_var_post, vv_var_pre - vv_var_post]
    if "unmix" in groups:
        if endmembers is None:
            raise ValueError("unmix group requires endmember spectra")
        frac_pre = unmix(pre, endmembers)
        frac_post = unmix(post, endmembers)
        layers += [*frac_pre, *frac_post, *(frac_post - frac_pre)]
    if "cloud" in groups:
        layers.append(ndi.uniform_filter(scl_bad.astype(np.float32), size=31))
    if "indices2" in groups:
        if kndvi_sigma is None:
            raise ValueError("indices2 group requires kNDVI sigma")
        bais_pre = _bais2(b4p, b6p, b7p, b8ap, b12p)
        bais_post = _bais2(b4q, b6q, b7q, b8aq, b12q)
        kndvi_pre = np.tanh(((b8ap - b4p) / (2.0 * kndvi_sigma)) ** 2)
        kndvi_post = np.tanh(((b8aq - b4q) / (2.0 * kndvi_sigma)) ** 2)
        layers += [bais_pre, bais_post, bais_pre - bais_post, kndvi_pre, kndvi_post, kndvi_pre - kndvi_post]
    if "pheno" in groups:
        corrected, delta, background = phenological_offset(dnbr, valid, mean15, landcover_idx)
        layers += [corrected, delta, background]

    stacked = np.stack([np.asarray(layer, dtype=np.float32) for layer in layers])
    stacked[np.isinf(stacked)] = 0.0
    return stacked, valid


def _bais2(b4: np.ndarray, b6: np.ndarray, b7: np.ndarray, b8a: np.ndarray, b12: np.ndarray) -> np.ndarray:
    diff = b12 - b8a
    inner = diff / (np.sqrt(np.abs(diff)) + EPS) + 1.0
    outer = np.sqrt(np.maximum(b6 * b7 * b8a / (b4 + EPS), 0.0))
    return np.clip(1.0 - outer * inner, -10.0, 10.0)
