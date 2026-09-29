"""ParamSpec / ParamSchema: types, bounds, choices, unknown names, round trip."""

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiago_ring_controller.blocks import (  # noqa: E402
    PRIMITIVE_SCHEMAS,
    ParamError,
    ParamSchema,
    ParamSpec,
    schema_for,
)


class ParamSpecTests(unittest.TestCase):
    def test_types_bounds_and_choices(self):
        count = ParamSpec("n", "int", 3, minimum=1, maximum=10)
        self.assertEqual(count.coerce(4.0), 4)
        for bad in (True, 0, 11, 2.5, "3"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                count.coerce(bad)
        weight = ParamSpec("w", "float", -0.6)
        self.assertEqual(weight.coerce(2), 2.0)
        with self.assertRaises(ValueError):
            weight.coerce(False)
        mode = ParamSpec("mode", "str", "once", choices=("once", "continuous"))
        with self.assertRaises(ValueError):
            mode.coerce("never")
        with self.assertRaises(ValueError):
            ParamSpec("bad_default", "int", 0, minimum=1)
        with self.assertRaises(ValueError):
            ParamSpec("bad_type", "complex", 0)
        self.assertEqual(
            mode.describe(), {"name": "mode", "type": "str", "default": "once", "choices": ["once", "continuous"]}
        )

    def test_dict_and_list_values_are_copied(self):
        params = ParamSpec("neuron_params", "dict", {})
        source = {"V_th": 0.8}
        value = params.coerce(source)
        value["V_th"] = 1.0
        self.assertEqual(source["V_th"], 0.8)
        with self.assertRaises(ValueError):
            params.coerce([1, 2])
        items = ParamSpec("items", "list", [])
        self.assertEqual(items.coerce((1, 2)), [1, 2])
        with self.assertRaises(ValueError):
            items.coerce("ab")


class ParamSchemaTests(unittest.TestCase):
    def test_resolve_reports_every_problem_at_once_and_defaults_missing(self):
        schema = schema_for("Ring")
        with self.assertRaises(ParamError) as raised:
            schema.resolve({"population_size": 2, "bogus": 1, "variant": "x"}, owner="r1")
        message = str(raised.exception)
        self.assertTrue(message.startswith("r1: "))
        self.assertIn("unknown parameter 'bogus'", message)
        self.assertIn("population_size: 2 is below the minimum 3", message)
        self.assertIn("variant: 'x' is not one of", message)
        resolved = schema.resolve({"population_size": 20})
        self.assertEqual(list(resolved), schema.names)
        self.assertEqual(resolved["population_size"], 20)
        self.assertEqual(resolved["variant"], "legacy")

    def test_json_round_trip_is_lossless_and_ordered(self):
        for name, schema in PRIMITIVE_SCHEMAS.items():
            with self.subTest(block=name):
                text = schema.dumps()
                self.assertEqual(list(json.loads(text)), schema.names)
                self.assertEqual(schema.loads(text), schema.defaults())
                self.assertEqual([item["name"] for item in schema.describe()], schema.names)

    def test_registry_and_duplicates(self):
        self.assertEqual(
            list(PRIMITIVE_SCHEMAS),
            ["Ring", "FourierReadout", "Homeostasis", "Gain", "Encoder", "Decoder", "Joint", "Goal"],
        )
        with self.assertRaises(KeyError):
            schema_for("Nope")
        with self.assertRaises(ValueError):
            ParamSchema("Dup", [ParamSpec("a", "int", 1), ParamSpec("a", "int", 2)])
        self.assertIn("mode", schema_for("Encoder"))
        self.assertEqual(schema_for("Encoder")["mode"].choices, ("once", "continuous", "corrective"))


if __name__ == "__main__":
    unittest.main()
