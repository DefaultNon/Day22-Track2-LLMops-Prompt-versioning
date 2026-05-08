import re
import json
from guardrails import Guard, OnFailAction, Validator, register_validator
from guardrails.validator_base import PassResult, FailResult

# ── 1. PII Detector Validator ─────────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """Detects and redacts PII using regex."""
    
    PII_PATTERNS = {
        "EMAIL":       r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",
        "PHONE":       r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}",
        "SSN":         r"\d{3}-\d{2}-\d{4}",
        "CREDIT_CARD": r"(?:\d{4}[-\s]?){3}\d{4}",
    }

    def validate(self, value: str, metadata: dict):
        redacted_text = value
        found_pii = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"  [DEBUG] Found {len(found_pii)} PII items. Redacting...")
            return FailResult(error_message="PII detected", fix_value=redacted_text)
        return PassResult()

# ── 2. JSON Formatter Validator ───────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """Validates and auto-repairs malformed JSON strings."""

    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        # Remove markdown fences (anywhere they might be)
        text = re.sub(r'```(?:json)?', '', text)
        text = re.sub(r'```', '', text)
        text = text.strip()
        # Single quotes -> double quotes (simple replacement)
        text = text.replace("'", '"')
        # Remove trailing commas before closing braces
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text

    def validate(self, value: str, metadata: dict):
        # 1. Try raw parse
        try:
            parsed = json.loads(value)
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            pass

        # 2. Try repair
        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            print(f"  [DEBUG] JSON repaired successfully.")
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            print(f"  [DEBUG] JSON repair failed: {e}")
            return FailResult(
                error_message=f"Invalid JSON: {e}",
                fix_value=json.dumps({"error": "unrecoverable", "raw": value})
            )

# ── 3. Demo Functions ────────────────────────────────────────────────────────
def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",       "Contact John at john.doe@example.com for details."),
        ("Phone",       "Call our support line at (555) 867-5309."),
        ("SSN",         "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",   "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",       "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")

def demo_json_guard():
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",        '{"name": "Alice", "age": 30}'),
        ("Markdown fences",   '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",     "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",    '{"key": "value",}'),
        ("Truly invalid",     "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "[PASS]" if result.validation_passed else "[FAIL]"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {str(result.validated_output).replace('\n', ' ')[:60]}")

def main():
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)
    demo_pii_guard()
    demo_json_guard()
    print("\n[OK] Step 4 complete!")

if __name__ == "__main__":
    main()
