"""Shared vocabularies: raw file columns, sensor names, and the three decision actions."""
from __future__ import annotations

# NASA's column order: unit, cycle, 3 operating settings, 21 sensors.
RAW_COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
OP_COLUMNS = ["op1", "op2", "op3"]

# Symbol, plain description, unit (Saxena et al., 2008).
SENSOR_INFO: dict[str, tuple[str, str, str]] = {
    "s1": ("T2", "Total temperature at fan inlet", "degR"),
    "s2": ("T24", "Total temperature at LPC outlet", "degR"),
    "s3": ("T30", "Total temperature at HPC outlet", "degR"),
    "s4": ("T50", "Total temperature at LPT outlet", "degR"),
    "s5": ("P2", "Pressure at fan inlet", "psia"),
    "s6": ("P15", "Total pressure in bypass duct", "psia"),
    "s7": ("P30", "Total pressure at HPC outlet", "psia"),
    "s8": ("Nf", "Physical fan speed", "rpm"),
    "s9": ("Nc", "Physical core speed", "rpm"),
    "s10": ("epr", "Engine pressure ratio (P50/P2)", "-"),
    "s11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "s12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "s13": ("NRf", "Corrected fan speed", "rpm"),
    "s14": ("NRc", "Corrected core speed", "rpm"),
    "s15": ("BPR", "Bypass ratio", "-"),
    "s16": ("farB", "Burner fuel-air ratio", "-"),
    "s17": ("htBleed", "Bleed enthalpy", "-"),
    "s18": ("Nf_dmd", "Demanded fan speed", "rpm"),
    "s19": ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    "s20": ("W31", "HPT coolant bleed", "lbm/s"),
    "s21": ("W32", "LPT coolant bleed", "lbm/s"),
}


def sensor_name(sensor: str) -> str:
    return SENSOR_INFO.get(sensor, (sensor, sensor, ""))[0]


def sensor_description(sensor: str) -> str:
    return SENSOR_INFO.get(sensor, (sensor, sensor, ""))[1]


def sensor_label(sensor: str) -> str:
    return f"{sensor_name(sensor)} ({sensor})"


# The decision engine's whole vocabulary: inspect, clear, or refuse to call.
ACTIONS = ("INSPECT", "NO_INSPECTION", "ABSTAIN")
ALL_ACTIONS = ACTIONS
# The only action the safety reviewer is allowed to escalate.
CLEARING_ACTIONS = ("NO_INSPECTION",)

ACTION_LABELS = {
    "INSPECT": "Inspect",
    "NO_INSPECTION": "No inspection needed",
    "ABSTAIN": "Abstain - engineer review",
}
