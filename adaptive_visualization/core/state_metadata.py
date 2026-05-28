"""State metadata shared by the OpenLayers adaptive visualization app."""

from __future__ import annotations

STATE_ROWS = [
    ("01", "AL", "Alabama"),
    ("02", "AK", "Alaska"),
    ("04", "AZ", "Arizona"),
    ("05", "AR", "Arkansas"),
    ("06", "CA", "California"),
    ("08", "CO", "Colorado"),
    ("09", "CT", "Connecticut"),
    ("10", "DE", "Delaware"),
    ("11", "DC", "District of Columbia"),
    ("12", "FL", "Florida"),
    ("13", "GA", "Georgia"),
    ("15", "HI", "Hawaii"),
    ("16", "ID", "Idaho"),
    ("17", "IL", "Illinois"),
    ("18", "IN", "Indiana"),
    ("19", "IA", "Iowa"),
    ("20", "KS", "Kansas"),
    ("21", "KY", "Kentucky"),
    ("22", "LA", "Louisiana"),
    ("23", "ME", "Maine"),
    ("24", "MD", "Maryland"),
    ("25", "MA", "Massachusetts"),
    ("26", "MI", "Michigan"),
    ("27", "MN", "Minnesota"),
    ("28", "MS", "Mississippi"),
    ("29", "MO", "Missouri"),
    ("30", "MT", "Montana"),
    ("31", "NE", "Nebraska"),
    ("32", "NV", "Nevada"),
    ("33", "NH", "New Hampshire"),
    ("34", "NJ", "New Jersey"),
    ("35", "NM", "New Mexico"),
    ("36", "NY", "New York"),
    ("37", "NC", "North Carolina"),
    ("38", "ND", "North Dakota"),
    ("39", "OH", "Ohio"),
    ("40", "OK", "Oklahoma"),
    ("41", "OR", "Oregon"),
    ("42", "PA", "Pennsylvania"),
    ("44", "RI", "Rhode Island"),
    ("45", "SC", "South Carolina"),
    ("46", "SD", "South Dakota"),
    ("47", "TN", "Tennessee"),
    ("48", "TX", "Texas"),
    ("49", "UT", "Utah"),
    ("50", "VT", "Vermont"),
    ("51", "VA", "Virginia"),
    ("53", "WA", "Washington"),
    ("54", "WV", "West Virginia"),
    ("55", "WI", "Wisconsin"),
    ("56", "WY", "Wyoming"),
    ("72", "PR", "Puerto Rico"),
]

STATE_CODE_TO_NAME = {code: name for _, code, name in STATE_ROWS}
STATE_NAME_TO_CODE = {name: code for _, code, name in STATE_ROWS}
STATE_FIPS_TO_CODE = {fips: code for fips, code, _ in STATE_ROWS}
STATE_CODE_TO_FIPS = {code: fips for fips, code, _ in STATE_ROWS}


def build_state_payload(available_codes: list[str] | None = None) -> dict:
    """Return a JSON-friendly state metadata payload for the frontend."""
    allowed = set(available_codes or STATE_CODE_TO_NAME.keys())
    rows = [
        {
            "fips": fips,
            "code": code,
            "name": name,
        }
        for fips, code, name in STATE_ROWS
        if code in allowed
    ]
    return {
        "rows": rows,
        "codeToName": {code: STATE_CODE_TO_NAME[code] for code in allowed},
        "nameToCode": {
            STATE_CODE_TO_NAME[code]: code
            for code in allowed
        },
        "fipsToCode": {
            STATE_CODE_TO_FIPS[code]: code
            for code in allowed
            if code in STATE_CODE_TO_FIPS
        },
    }
