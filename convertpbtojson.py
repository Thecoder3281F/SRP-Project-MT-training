"""Convert ORD parquet datasets into cleaner CSV files.

This script uses the ORD schema directly. It reads each parquet dataset as an
ORD Dataset, iterates through the Reaction messages, and writes a structured
CSV with readable schema fields instead of raw protobuf bytes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from google.protobuf.json_format import MessageToDict
from ord_schema.parquet import iter_reactions, load_dataset


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    slug = _SLUG_RE.sub("_", value.lower()).strip("_")
    return slug or "field"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ORD parquet datasets to CSV using the ORD schema")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", type=Path, help="Input .parquet file")
    source_group.add_argument("--input-root", type=Path, help="Root directory to scan for .parquet files")
    parser.add_argument("--output", type=Path, help="Output CSV file for single-file mode")
    parser.add_argument("--output-root", type=Path, help="Root directory for CSV outputs in recursive mode")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories when using --input-root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output CSV if it exists")
    return parser.parse_args()


def find_parquet_files(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(path for path in root.rglob("*.parquet") if path.is_file())
    return sorted(path for path in root.glob("*.parquet") if path.is_file())


def output_path_for_input(input_path: Path, input_root: Path, output_root: Path | None) -> Path:
    relative_path = input_path.relative_to(input_root)
    csv_name = relative_path.with_suffix(".csv").name
    if output_root is None:
        return input_path.with_name(csv_name)
    return output_root / relative_path.parent / csv_name


def get_identifier_value(identifiers: list[dict[str, object]], identifier_type: str) -> str:
    for identifier in identifiers:
        if identifier.get("type") == identifier_type:
            value = identifier.get("value")
            if value is not None:
                return str(value)
    return ""


def join_values(values: list[object]) -> str:
    return " | ".join(str(value) for value in values if value not in (None, ""))


def summarize_component(component: dict[str, object]) -> dict[str, str]:
    identifiers = component.get("identifiers", [])
    amount = component.get("amount", {})
    moles = amount.get("moles", {}) if isinstance(amount, dict) else {}

    return {
        "smiles": get_identifier_value(identifiers, "SMILES") if isinstance(identifiers, list) else "",
        "amount_moles": str(moles.get("value", "")) if isinstance(moles, dict) else "",
        "amount_units": str(moles.get("units", "")) if isinstance(moles, dict) else "",
        "reaction_role": str(component.get("reaction_role", "")),
        "is_limiting": str(component.get("is_limiting", "")),
    }


def summarize_role(role_name: str, role_value: dict[str, object]) -> dict[str, str]:
    components = role_value.get("components", []) if isinstance(role_value, dict) else []
    summarized = [summarize_component(component) for component in components if isinstance(component, dict)]

    prefix = f"input_{slugify(role_name)}"
    return {
        f"{prefix}_smiles": join_values([item["smiles"] for item in summarized]),
        f"{prefix}_moles": join_values([item["amount_moles"] for item in summarized]),
        f"{prefix}_amount_units": join_values([item["amount_units"] for item in summarized]),
        f"{prefix}_reaction_roles": join_values([item["reaction_role"] for item in summarized]),
        f"{prefix}_is_limiting": join_values([item["is_limiting"] for item in summarized]),
    }


def reaction_to_row(reaction_id: str, reaction) -> dict[str, str]:
    reaction_dict = MessageToDict(
        reaction,
        preserving_proto_field_name=True,
        use_integers_for_enums=False,
    )

    row: dict[str, str] = {
        "reaction_id": str(reaction_dict.get("reaction_id", reaction_id)),
        "reaction_index": "",
        "reaction_type": "",
        "procedure_details": "",
        "temperature_celsius": "",
        "temperature_precision_celsius": "",
        "publication_url": "",
        "experiment_start": "",
        "experimenter_organization": "",
        "record_created_time": "",
        "record_created_person_name": "",
        "record_created_person_organization": "",
        "record_created_person_email": "",
        "record_modified_details": "",
        "product_smiles": "",
        "product_yield_percent": "",
        "product_is_desired": "",
    }

    identifiers = reaction_dict.get("identifiers", [])
    if isinstance(identifiers, list):
        for identifier in identifiers:
            if not isinstance(identifier, dict):
                continue
            identifier_type = identifier.get("type")
            value = str(identifier.get("value", ""))
            details = str(identifier.get("details", ""))
            if identifier_type == "CUSTOM" and details == "reaction index":
                row["reaction_index"] = value
            elif identifier_type == "REACTION_TYPE":
                row["reaction_type"] = value

    notes = reaction_dict.get("notes", {})
    if isinstance(notes, dict):
        row["procedure_details"] = str(notes.get("procedure_details", ""))

    conditions = reaction_dict.get("conditions", {})
    if isinstance(conditions, dict):
        temperature = conditions.get("temperature", {})
        if isinstance(temperature, dict):
            setpoint = temperature.get("setpoint", {})
            if isinstance(setpoint, dict):
                row["temperature_celsius"] = str(setpoint.get("value", ""))
                row["temperature_precision_celsius"] = str(setpoint.get("precision", ""))

    provenance = reaction_dict.get("provenance", {})
    if isinstance(provenance, dict):
        row["publication_url"] = str(provenance.get("publication_url", ""))
        row["experiment_start"] = str(provenance.get("experiment_start", {}).get("value", "")) if isinstance(provenance.get("experiment_start", {}), dict) else ""
        experimenter = provenance.get("experimenter", {})
        if isinstance(experimenter, dict):
            row["experimenter_organization"] = str(experimenter.get("organization", ""))

        record_created = provenance.get("record_created", {})
        if isinstance(record_created, dict):
            row["record_created_time"] = str(record_created.get("time", {}).get("value", "")) if isinstance(record_created.get("time", {}), dict) else ""
            person = record_created.get("person", {})
            if isinstance(person, dict):
                row["record_created_person_name"] = str(person.get("name", ""))
                row["record_created_person_organization"] = str(person.get("organization", ""))
                row["record_created_person_email"] = str(person.get("email", ""))

        record_modified = provenance.get("record_modified", [])
        if isinstance(record_modified, list):
            row["record_modified_details"] = join_values([
                item.get("details", "")
                for item in record_modified
                if isinstance(item, dict)
            ])

    inputs = reaction_dict.get("inputs", {})
    if isinstance(inputs, dict):
        for role_name, role_value in inputs.items():
            if isinstance(role_value, dict):
                row.update(summarize_role(str(role_name), role_value))

    outcomes = reaction_dict.get("outcomes", [])
    if isinstance(outcomes, list):
        product_smiles = []
        product_yields = []
        desired_flags = []
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            products = outcome.get("products", [])
            if not isinstance(products, list):
                continue
            for product in products:
                if not isinstance(product, dict):
                    continue
                identifiers = product.get("identifiers", [])
                if isinstance(identifiers, list):
                    product_smiles.append(get_identifier_value(identifiers, "SMILES"))
                measurements = product.get("measurements", [])
                if isinstance(measurements, list):
                    for measurement in measurements:
                        if isinstance(measurement, dict) and measurement.get("type") == "YIELD":
                            percentage = measurement.get("percentage", {})
                            if isinstance(percentage, dict):
                                product_yields.append(str(percentage.get("value", "")))
                desired_flags.append(str(product.get("is_desired_product", "")))

        row["product_smiles"] = join_values(product_smiles)
        row["product_yield_percent"] = join_values(product_yields)
        row["product_is_desired"] = join_values(desired_flags)

    return row


def convert_file(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {output_path}. Use --overwrite to replace it.")

    rows = []
    for reaction_id, reaction in iter_reactions(input_path):
        rows.append(reaction_to_row(reaction_id, reaction))

    if not rows:
        print(f"No reactions found in: {input_path}")
        return

    dataframe = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Wrote CSV to: {output_path}")


def main() -> None:
    args = parse_args()

    if args.input is not None:
        if args.output is None:
            raise ValueError("--output is required when using --input")
        convert_file(args.input, args.output, args)
        return

    if args.output is not None:
        raise ValueError("--output can only be used with --input")

    if args.input_root is None:
        raise ValueError("--input-root is required when not using --input")

    input_root = args.input_root
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    output_root = args.output_root or input_root
    input_files = find_parquet_files(input_root, args.recursive)
    if not input_files:
        print(f"No .parquet files found under: {input_root}")
        return

    for input_path in input_files:
        output_path = output_path_for_input(input_path, input_root, output_root)
        convert_file(input_path, output_path, args)


if __name__ == "__main__":
    main()